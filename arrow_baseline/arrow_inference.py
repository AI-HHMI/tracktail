"""
Run ARROW (https://github.com/ifradlin/arrow) multi-view tracking on all trials of a dataset split, with the queries
of a pose file, and save predictions in the same format as mvtracker_inference.py (so mvtracker_metrics.py can score
them with --prefix arrow_).

ARROW is RGB-only: it gets the images of the cameras in use, but no depth, no calibration and no 3D queries. The
protocol follows the ARROW paper's RGB-only adaptation of the MVTracker benchmark (arXiv 2610.01314, Sec. 4, App. C.1):
  * each track is queried with a 2D pixel: its ground-truth 3D position at the query frame projected into one source
    view, chosen uniformly (seeded) among the cameras in use that see it at that frame (the calibration is used only
    for this projection and for the evaluation targets);
  * each query is predicted at every frame of the trial, before and after the query frame; the 3D position of a
    timestep is the prediction at the first camera's target slot, in the first camera's coordinate system;
  * predicted visibility is the maximum over the cameras of the visibility logits;
  * predictions have an arbitrary scale: one scale per encoding (per scene when it fits in one window) is fit,
    by default as in the paper, Eq. 11 (median ||x_gt|| / median ||x_pred|| over valid, visible positions with finite
    predictions, in the first camera's frame), or with --scale-mode cameras from the ground-truth camera centers
    (ARROW's camera poses of the first timestep vs the calibration; uses no track ground truth). The scaled
    predictions are mapped to the world with the first camera's ground-truth extrinsics.

Long trials (cmupanoptic_3dgs: 150 frames, Point Odyssey: 499 frames) do not fit in one encoding. They are run in
windows of --window timesteps (all cameras in use); a track whose query frame is outside a window is queried from its
source image, which is added to that window's encoding as an anchor image (ARROW is order-invariant, with no
temporal encoding, so any image set works). At most --max-anchors anchor images are added per encoding; more anchors
are split across several encodings of the same window. The cameras are static, so the first camera's frame is the same
coordinate system for every window; each encoding has its own scale.

Outputs:  <output_root>/<dataset>/<session>_<trial>.npz   (fields as in mvtracker_inference.py)

Run with ARROW's environment (cd ../arrow && uv sync):
    ../arrow/.venv/bin/python arrow_inference.py --dataset-root ... --output-root ... --datasets dex_ycb \\
        --cam-names view_00 view_01 view_02 view_03 --pose-file pose3d_cached.npz --query-first
"""

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from arrow import ARROW
from arrow.data.transforms import prepare_images
from arrow.queries import pixel_grid
from arrow.tasks.pose import _weighted_rigid
from arrow.types import Queries


# ── dataset helpers ───────────────────────────────────────────────────────────

def get_dirs(path):
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def natural_key(name):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def load_cameras(metadata_path):
    """Camera names (sorted as in mvtracker_inference.py) and per-camera K (3,3), world-to-camera E (4,4), size."""
    with open(metadata_path) as f:
        meta = yaml.safe_load(f)
    if meta.get("moving_cams", False):
        raise ValueError("moving cameras are not supported")
    names = list(meta["intrinsic_matrices"].keys())
    names = sorted(names, key=int) if all(n.isdigit() for n in names) else sorted(names)
    K = np.stack([np.asarray(meta["intrinsic_matrices"][n], dtype=np.float64) for n in names])
    E = []
    for n in names:
        e = np.asarray(meta["extrinsic_matrices"][n], dtype=np.float64)
        if e.shape == (3, 4):
            e = np.concatenate([e, [[0, 0, 0, 1]]])
        E.append(e)
    return names, K, np.stack(E)


def compute_query_first(coords, vis_any):
    """First frame with finite coords and visible in >=1 used camera (as tracktail's compute_query_first).
    coords (T, N, 3), vis_any (T, N) bool or None. Returns query_time (N,), valid (N,)."""
    good = np.all(np.isfinite(coords), axis=-1)
    if vis_any is not None:
        good &= vis_any
    return np.argmax(good, axis=0).astype(np.int64), good.any(axis=0)


def project(X, K, E):
    """World points (N,3) -> pixels (N,2) and camera depth (N,)."""
    xc = X @ E[:3, :3].T + E[:3, 3]
    uv = xc[:, :2] / xc[:, 2:3]
    return uv * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]], xc[:, 2]


# ── ARROW encoding / decoding ─────────────────────────────────────────────────

class ImageCache:
    """Prepared images (3, H, W) float in [0, 1], per (frame, camera index), with their original->input transforms."""

    def __init__(self, img_dirs, max_side, size=512):
        self.files = [sorted(os.listdir(d), key=natural_key) for d in img_dirs]
        self.dirs, self.max_side, self.size, self.cache = img_dirs, max_side, size, {}

    def n_frames(self):
        return min(len(f) for f in self.files)

    def get(self, t, c):
        key = (t, c)
        if key not in self.cache:
            if len(self.cache) >= self.size:
                self.cache.pop(next(iter(self.cache)))
            with Image.open(os.path.join(self.dirs[c], self.files[c][t])) as im:
                im = im.convert("RGB")
            info = {}
            tensor = prepare_images([im], [info], max_side=self.max_side)[0]
            self.cache[key] = (tensor, np.asarray(info["original_to_input"], dtype=np.float64))
        return self.cache[key]


def camera_scale(model, encoded, V, E_used, decode_chunk_size, grid_size=32):
    """Scale from the camera centers: ARROW's poses of frames 0..V-1 (timestep 0 of the window, frame 0 = camera 0)
    against the ground-truth centers in camera 0's frame. Median over cameras of |c_gt| / |c_pred|."""
    _, _, _, H, W = encoded.images.shape
    xy = pixel_grid(H, W, size=grid_size, device=encoded.device)
    n = len(xy)
    frames = torch.arange(V, device=encoded.device).repeat_interleave(n)
    q = Queries(torch.cat([xy.repeat(V, 1)] * 2), torch.cat([frames] * 2), torch.cat([frames] * 2),
                torch.cat([frames, torch.zeros_like(frames)]))
    with torch.inference_mode():
        p = model.decode(encoded, q, decode_chunk_size=decode_chunk_size)
    pts = p.points[0].float().cpu().unflatten(0, (2, V, n))
    conf = p.point_confidence[0].float().cpu().unflatten(0, (2, V, n))
    vis = p.visibility_logits[0].float().cpu().unflatten(0, (2, V, n))
    E0inv = np.linalg.inv(E_used[0])
    ratios = []
    for v in range(1, V):
        own, shared = pts[0, v], pts[1, v]
        w = conf[0, v] * conf[1, v]
        ok = torch.isfinite(own).all(-1) & torch.isfinite(shared).all(-1) & (vis[0, v] > 0) & (vis[1, v] > 0)
        idx = ok.nonzero(as_tuple=True)[0]
        if len(idx) < 8:
            continue
        idx = idx[w[idx] >= torch.quantile(w[idx], 0.5)]
        rigid = _weighted_rigid(shared[idx], own[idx], w[idx].pow(2))
        if rigid is None:
            continue
        c_pred = rigid[1].numpy()                        # camera v center in camera 0's frame (predicted)
        c_gt = (E_used[0] @ np.linalg.inv(E_used[v]))[:3, 3]   # camera v center in camera 0's frame (GT)
        if np.linalg.norm(c_pred) > 1e-8:
            ratios.append(np.linalg.norm(c_gt) / np.linalg.norm(c_pred))
    del E0inv
    return float(np.median(ratios)) if ratios else None


def run_encoding(model, images, src_frame, xy, targets_t, V, device, decode_chunk_size):
    """Encode images [F,3,H,W] (frames 0..V*len(targets_t)-1 are the window, t-major, anchors after), decode each
    query at every window frame with reference frame 0. Returns points (N, Tw, 3) at camera 0's slot, visibility
    logits (N, Tw, V), and the encoding."""
    Tw = len(targets_t)
    F_win = Tw * V
    imgs = images.unsqueeze(0).to(device)
    with torch.inference_mode():
        encoded = model.encode(imgs, anchor_frame=0)
        N = len(xy)
        xy_t = torch.as_tensor(xy, dtype=torch.float32, device=device)
        # seed-major order: query n, target f  ->  index n * F_win + f
        q = Queries(xy_t.repeat_interleave(F_win, 0),
                    torch.as_tensor(src_frame, device=device).repeat_interleave(F_win),
                    torch.arange(F_win, device=device).repeat(N),
                    0)
        p = model.decode(encoded, q, decode_chunk_size=decode_chunk_size)
    pts = p.points[0].float().cpu().numpy().reshape(N, Tw, V, 3)[:, :, 0]
    vis = p.visibility_logits[0].float().cpu().numpy().reshape(N, Tw, V)
    return pts, vis, encoded


def fit_scale_gt(pred_cam0, gt_cam0, mask):
    """Paper Eq. 11: median ||x_gt|| / median ||x_pred|| over masked positions with finite predictions."""
    ok = mask & np.isfinite(pred_cam0).all(-1) & np.isfinite(gt_cam0).all(-1)
    if ok.sum() == 0:
        return None
    den = np.median(np.linalg.norm(pred_cam0[ok], axis=-1))
    return float(np.median(np.linalg.norm(gt_cam0[ok], axis=-1)) / den) if den > 0 else None


# ── per-trial processing ──────────────────────────────────────────────────────

def process_trial(trial_path, out_path, model, device, args, dataset_name):
    metadata_path = os.path.join(trial_path, "metadata.yaml")
    pose_path = args.pose_file if os.path.isabs(args.pose_file) else os.path.join(trial_path, args.pose_file)
    for p in (metadata_path, pose_path):
        if not os.path.exists(p):
            print(f"    [skip] missing {p}")
            return
    if os.path.exists(out_path) and not args.overwrite:
        print("    [skip] already done (--overwrite to redo)")
        return

    names_all, K_all, E_all = load_cameras(metadata_path)
    if args.cam_names:
        missing = [n for n in args.cam_names if n not in names_all]
        if missing:
            print(f"    [skip] requested cameras not in metadata: {missing}")
            return
        cam_idx = [names_all.index(n) for n in args.cam_names]
    else:
        cam_idx = list(range(len(names_all)))
    cam_names = [names_all[i] for i in cam_idx]
    K_used, E_used = K_all[cam_idx], E_all[cam_idx]
    V = len(cam_idx)

    img_dirs = [os.path.join(trial_path, "img", n) for n in cam_names]
    if not all(os.path.isdir(d) for d in img_dirs):
        print("    [skip] missing img/<camera> folders")
        return
    images = ImageCache(img_dirs, args.max_side, size=max(512, 4 * V * args.window))

    pose = np.load(pose_path)
    coords_gt, vis_gt_raw = pose["pose"], (pose["vis"] if "vis" in pose.files else None)
    S, T_full, Kp, _ = coords_gt.shape
    if vis_gt_raw is not None and vis_gt_raw.shape[-1] != len(names_all):
        raise ValueError(f"{pose_path}: vis has {vis_gt_raw.shape[-1]} columns, metadata {len(names_all)} cameras")
    T = min(T_full, images.n_frames())
    if args.n_frames is not None:
        T = min(T, args.n_frames)
    frame_numbers = np.arange(T, dtype=np.int64)

    # tracks (S*K) in mvtracker_inference.py's order: subject-major
    coords_flat = coords_gt.transpose(0, 2, 1, 3).reshape(S * Kp, T_full, 3)[:, :T]          # (M, T, 3)
    vis_cams = (vis_gt_raw[..., cam_idx].transpose(0, 2, 1, 3).reshape(S * Kp, T_full, V)[:, :T] > 0.5
                if vis_gt_raw is not None else None)                                        # (M, T, V)
    if args.query_first:
        qt, valid = compute_query_first(coords_flat.transpose(1, 0, 2),
                                        vis_cams.any(-1).T if vis_cams is not None else None)
    else:
        qt = np.zeros(len(coords_flat), dtype=np.int64)
        valid = np.isfinite(coords_flat[:, 0]).all(-1)
        if vis_cams is not None:
            valid &= vis_cams[:, 0].any(-1)
    idx = np.nonzero(valid)[0]
    if len(idx) == 0:
        print("    [skip] no valid queries")
        return
    N = len(idx)
    qt = qt[idx]
    q3d = coords_flat[idx, qt]                                                               # (N, 3)

    # source view: uniform among the used cameras that see the point at its query frame (all cameras if no vis)
    rng = np.random.default_rng(args.query_seed)
    src_view = np.empty(N, dtype=np.int64)
    src_xy_orig = np.empty((N, 2))
    for n in range(N):
        cand = np.nonzero(vis_cams[idx[n], qt[n]])[0] if vis_cams is not None else np.arange(V)
        if len(cand) == 0:
            cand = np.arange(V)
        # prefer cameras where the projection is in front of the camera
        uvs = [project(q3d[n:n + 1], K_used[c], E_used[c]) for c in cand]
        front = [c for c, (_, z) in zip(cand, uvs) if z[0] > 0]
        c = rng.choice(front if front else cand)
        src_view[n] = c
        src_xy_orig[n] = project(q3d[n:n + 1], K_used[c], E_used[c])[0][0]

    # predictions in camera 0's frame (before scaling) and visibility logits, per window
    pred_cam0 = np.full((N, T, 3), np.nan, dtype=np.float64)
    vis_logit = np.full((N, T, V), -10.0, dtype=np.float32)
    gt_cam0 = coords_flat[idx] @ E_used[0][:3, :3].T + E_used[0][:3, 3]                      # (N, T, 3)
    vis_true_any = vis_cams[idx].any(-1) if vis_cams is not None else np.isfinite(gt_cam0).all(-1)
    scales, n_enc, t0 = [], 0, time.time()

    # round(T / window) windows of (nearly) equal length, so that no window is very short (25 frames: one window)
    n_win = max(1, round(T / args.window))
    bounds = np.linspace(0, T, n_win + 1).round().astype(int)
    for w0, w1 in zip(bounds[:-1], bounds[1:]):
        w0, w1 = int(w0), int(w1)
        win_t = list(range(w0, w1))
        Tw = len(win_t)
        inside = (qt >= w0) & (qt < w0 + Tw)
        # anchors: source images outside the window, grouped in chunks of max_anchors
        outside_src = sorted(set((int(qt[n]), int(src_view[n])) for n in np.nonzero(~inside)[0]))
        chunks = [outside_src[i:i + args.max_anchors] for i in range(0, len(outside_src), args.max_anchors)] or [[]]
        for ci, anchors in enumerate(chunks):
            anchor_pos = {a: Tw * V + i for i, a in enumerate(anchors)}
            sel = [n for n in range(N) if (ci == 0 and inside[n]) or (int(qt[n]), int(src_view[n])) in anchor_pos]
            if not sel:
                continue
            sel = np.asarray(sel)
            frame_keys = [(t, v) for t in win_t for v in range(V)] + list(anchors)
            loaded = [images.get(t, v) for t, v in frame_keys]
            imgs = torch.stack([x[0] for x in loaded])
            src_frame, xy = [], []
            for n in sel:
                key = (int(qt[n]), int(src_view[n]))
                f = (key[0] - w0) * V + key[1] if inside[n] else anchor_pos[key]
                A = loaded[f][1]
                xy.append(A[:2, :2] @ src_xy_orig[n] + A[:2, 2])
                src_frame.append(f)
            H, W = imgs.shape[-2:]
            xy = np.clip(np.asarray(xy), -0.5, [W - 0.5, H - 0.5])
            pts, vis, encoded = run_encoding(model, imgs, np.asarray(src_frame), xy, win_t, V, device,
                                             args.decode_chunk_size)
            if args.scale_mode == "cameras":
                s = camera_scale(model, encoded, V, E_used, args.decode_chunk_size) if V > 1 else None
            else:
                s = fit_scale_gt(pts, gt_cam0[sel][:, win_t], vis_true_any[sel][:, win_t])
            del encoded
            if s is None:
                print(f"    [warn] no scale for window {w0} chunk {ci}; using 1")
                s = 1.0
            scales.append(s)
            pred_cam0[sel[:, None], np.asarray(win_t)[None]] = pts * s
            vis_logit[sel[:, None], np.asarray(win_t)[None]] = vis
            n_enc += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # camera 0 -> world
    E0inv = np.linalg.inv(E_used[0])
    coords_pred_world = pred_cam0 @ E0inv[:3, :3].T + E0inv[:3, 3]                           # (N, T, 3)
    vis_pred = vis_logit.max(-1)                                                             # (N, T)

    coords_true = coords_flat[idx].transpose(1, 0, 2)[None]                                  # (1, T, N, 3)
    vis_true = vis_true_any.T[None, :, :, None]                                              # (1, T, N, 1)
    valid_mask = np.zeros(S * Kp, dtype=bool)
    valid_mask[idx] = True
    save = dict(
        coords_pred=coords_pred_world.transpose(1, 0, 2)[None].astype(np.float32),            # (1, T, N, 3)
        vis_pred=vis_pred.T[None, :, :, None].astype(np.float32),                            # (1, T, N, 1) logits
        conf_pred=vis_pred.T[None].astype(np.float32),                                       # (1, T, N)
        vis_prob=(1 / (1 + np.exp(-vis_pred.T))).astype(np.float32),                          # (T, N)
        coords_true=coords_true.astype(np.float32),
        vis_true=vis_true.astype(bool),
        query_times=qt.astype(np.int32),
        query_views=src_view.astype(np.int32),                                               # index into cam_names_used
        query_xy=src_xy_orig.astype(np.float32),                                             # original-image pixels
        scales=np.asarray(scales, dtype=np.float32),
        frame_numbers=frame_numbers,
        valid_mask=valid_mask,
        cam_names_used=np.array(cam_names, dtype=str),
        cam_indices_used=np.array(cam_idx, dtype=np.int32),
        metadata_path=np.array(metadata_path, dtype=str),
        video_paths=np.array(img_dirs, dtype=str),                                           # for render_video.py
        start_frame=np.array(0, dtype=np.int64),
    )
    if vis_cams is not None:
        save["vis_true_cams"] = vis_cams[idx].transpose(1, 0, 2)[None].astype(bool)          # (1, T, N, V)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **save)
    print(f"    saved: T={T} N={N} cams={cam_names} encodings={n_enc} "
          f"scale median={np.median(scales):.3g} ({time.time() - t0:.0f}s)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", default="ifradlin/arrow", help="Hub ID or local model directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cam-names", nargs="+", default=None, help="cameras to use, in order (first = reference)")
    p.add_argument("--pose-file", default="pose3d.npz")
    p.add_argument("--query-first", action="store_true",
                   help="query each track at its first frame that is finite and visible in a used camera")
    p.add_argument("--query-seed", type=int, default=0, help="seed of the source-view choice")
    p.add_argument("--n-frames", type=int, default=None)
    p.add_argument("--window", type=int, default=24, help="timesteps per encoding (all cameras)")
    p.add_argument("--max-anchors", type=int, default=24,
                   help="max source images added to a window's encoding for queries outside it")
    p.add_argument("--max-side", type=int, default=504)
    p.add_argument("--scale-mode", choices=["gt-tracks", "cameras"], default="gt-tracks",
                   help="gt-tracks: paper Eq. 11 per encoding; cameras: from the GT camera centers")
    p.add_argument("--decode-chunk-size", type=int, default=100_000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = ARROW.from_pretrained(args.checkpoint, device=str(device))
    for dataset_name in args.datasets:
        split_path = os.path.join(args.dataset_root, dataset_name, args.split)
        trials = [(s, t) for s in get_dirs(split_path) for t in get_dirs(os.path.join(split_path, s))]
        for session, trial in tqdm(trials, desc=dataset_name, unit="trial"):
            tqdm.write(f"  {session}/{trial}")
            out_path = os.path.join(args.output_root, dataset_name, f"{session}_{trial}.npz")
            process_trial(os.path.join(split_path, session, trial), out_path, model, device, args, dataset_name)
    print("Done.")


if __name__ == "__main__":
    main()
