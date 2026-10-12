"""
Run CoTracker3 inference on all trials in a dataset split.

For each trial, projects 3D query points (from pose3d.npz) into each camera's
2D pixel space, then runs CoTracker3 on that camera's video independently.

Outputs are saved to:
    trial_path/cotracker/cam_name.npz

Each .npz contains:
    tracks        (T, N, 2)  float32   pixel (x, y) per frame per keypoint
    visibility    (T, N)     bool      CoTracker confidence mask
    queries_2d    (N, 2)     float32   initial 2D query pixel (x, y) at start_frame
    frame_numbers (T,)       int64     absolute frame indices in the source video
    cam_name      str
    video_size    (2,)       int32     (W, H) of the video fed to CoTracker
    valid_mask    (S*K,)     bool      which flattened (subject, kpt) entries had
                                       finite 3D coords and are included as queries
    start_frame   int

Example
-------
    python run_cotracker_inference.py \\
        --dataset-path /data/myproject \\
        --split test \\
        --checkpoint /weights/scaled_offline.pth \\
        --device cuda:0 \\
        --n-frames 300 \\
        --window-len 60

Weights can be downloaded from:
    https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
"""

import os
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

from cotracker.predictor import CoTrackerPredictor

# -- reuse helpers from your existing pipeline ---------------------------
from posetail.datasets.utils import get_dirs
from posetail.posetail.cube import project_points_torch
from posetail.inference.inference_utils import (
    load_camera_group_from_metadata,   # returns formatted list-of-dicts
    build_video_readers,
    load_multiview_clip,
)
# ------------------------------------------------------------------------


# ── helpers ──────────────────────────────────────────────────────────────

def load_query_points(pose_path: str, start_frame: int = 0):
    """
    Load 3D query points from pose3d.npz.

    Returns
    -------
    query_3d : (N, 3) float32 tensor  – valid (finite) 3D coords at start_frame
    valid_mask : (S*K,) bool ndarray  – which flattened (subject × kpt) entries
                                        are included
    """
    data = np.load(pose_path)
    coords = data["pose"]                          # (S, T, K, 3)
    S, T, K, _ = coords.shape

    coords_at_start = coords[:, start_frame, :, :]  # (S, K, 3)
    coords_flat = coords_at_start.reshape(-1, 3)    # (S*K, 3)

    valid = np.all(np.isfinite(coords_flat), axis=1)
    query_3d = torch.as_tensor(coords_flat[valid], dtype=torch.float32)

    return query_3d, valid


def project_to_camera(camera_group, query_3d: torch.Tensor, cam_idx: int) -> torch.Tensor:
    """
    Project (N, 3) 3D world coords to pixel (x, y) for one camera.

    Returns
    -------
    (N, 2) float32 tensor  – pixel (x, y) coordinates
    """
    # project_points_torch returns (n_cams, N, 2)
    p2d = project_points_torch(camera_group, query_3d)
    return p2d[cam_idx]  # (N, 2)


def run_cotracker_single_camera(
    model: CoTrackerPredictor,
    frames: torch.Tensor,        # (T, H, W, 3) uint8 or float32 in [0,255]
    queries_2d: torch.Tensor,    # (N, 2) float32, pixel (x, y) at frame 0
    device: torch.device,
    query_batch_size: int | None = None,
):
    """
    Run CoTracker3 on one camera's video.

    If query_batch_size is set, queries are split into chunks and run
    independently (each chunk uses its own attention), then concatenated.
    This trades result quality for memory when N is large.

    Returns
    -------
    tracks     : (T, N, 2) float32  – pixel (x, y)
    visibility : (T, N)   bool
    """
    # frames may be uint8; ensure float32 in [0, 255]
    video = frames.to(dtype=torch.float32)          # (T, H, W, 3)
    video = video.permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, 3, H, W)
    video = video.to(device)

    N = queries_2d.shape[0]

    if query_batch_size is None or N <= query_batch_size:
        chunks = [queries_2d]
    else:
        chunks = list(torch.split(queries_2d, query_batch_size, dim=0))

    all_tracks = []
    all_vis = []
    for chunk in chunks:
        n = chunk.shape[0]
        t_col = torch.zeros(n, 1, device=device)
        queries = torch.cat([t_col, chunk.to(device)], dim=-1).unsqueeze(0)  # (1, n, 3)

        with torch.no_grad():
            tracks, visibility = model(video, queries=queries)
            # tracks:     (1, T, n, 2)
            # visibility: (1, T, n)  bool

        all_tracks.append(tracks.squeeze(0).cpu())
        all_vis.append(visibility.squeeze(0).cpu())
        torch.cuda.empty_cache()

    return torch.cat(all_tracks, dim=1), torch.cat(all_vis, dim=1)


TARGET_RES = 256  # resize-and-pad target matching posetail's inference resolution


def resize_and_pad(frames: np.ndarray, target: int = TARGET_RES):
    """
    Resize (T, H, W, 3) so the long edge == target, then zero-pad to (T, target, target, 3).
    Returns (padded, scale, pad_left, pad_top).
    """
    T, H, W, C = frames.shape
    scale = target / max(H, W)
    new_H = int(round(H * scale))
    new_W = int(round(W * scale))
    resized = np.stack([cv2.resize(f, (new_W, new_H)) for f in frames], axis=0)
    pad_top  = (target - new_H) // 2
    pad_left = (target - new_W) // 2
    padded = np.zeros((T, target, target, C), dtype=frames.dtype)
    padded[:, pad_top:pad_top + new_H, pad_left:pad_left + new_W] = resized
    return padded, scale, pad_left, pad_top


# ── per-trial processing ─────────────────────────────────────────────────

def process_trial(
    trial_path: str,
    outdir: str,
    model: CoTrackerPredictor,
    device: torch.device,
    start_frame: int = 0,
    n_frames: int | None = None,
    max_res: int | None = None,
    query_batch_size: int | None = None,
    overwrite: bool = False,
    pose_file: str | None = None,
):
    os.makedirs(outdir, exist_ok=True)

    metadata_path = os.path.join(trial_path, "metadata.yaml")
    if pose_file is not None:
        pose_path = pose_file if os.path.isabs(pose_file) else os.path.join(trial_path, pose_file)
    else:
        pose_path = os.path.join(trial_path, "pose3d.npz")

    if not os.path.exists(metadata_path):
        print(f"    [skip] missing metadata.yaml")
        return
    if not os.path.exists(pose_path):
        print(f"    [skip] missing pose3d.npz")
        return

    # Load formatted camera group (list of dicts) and camera names
    camera_group = load_camera_group_from_metadata(metadata_path, device="cpu")
    cam_names = [cam["name"] for cam in camera_group]

    # Skip if all outputs already exist
    if not overwrite:
        existing = [os.path.join(outdir, f"{n}.npz") for n in cam_names]
        if all(os.path.exists(p) for p in existing):
            print(f"    [skip] already done (--overwrite to redo)")
            return

    # Load 3D query points
    query_3d, valid_mask = load_query_points(pose_path, start_frame=start_frame)
    if query_3d.shape[0] == 0:
        print(f"    [skip] no finite 3D coords at frame {start_frame}")
        return

    # Locate video / image data
    img_path = os.path.join(trial_path, "img")
    vid_path = os.path.join(trial_path, "vid")

    if os.path.exists(img_path) and os.listdir(img_path):
        video_paths = [os.path.join(img_path, n) for n in cam_names]
    elif os.path.exists(vid_path):
        video_paths = [os.path.join(vid_path, f"{n}.mp4") for n in cam_names]
    else:
        print(f"    [skip] no img/ or vid/ folder")
        return

    readers, lengths = build_video_readers(video_paths)
    max_available = min(lengths)

    end_frame = min(
        start_frame + n_frames if n_frames is not None else max_available,
        max_available,
    )
    actual_n = end_frame - start_frame
    frame_numbers = np.arange(start_frame, end_frame, dtype=np.int64)

    # Load all frames at once (shape: list of (T, H, W, 3) tensors per camera)
    # For very long sequences (>500 frames at high res) this may OOM; reduce
    # --n-frames or add --max-res to manage memory.
    views, _ = load_multiview_clip(readers, start_frame, actual_n)
    del readers

    for cam_idx, cam_name in enumerate(tqdm(cam_names, desc="cameras", leave=False)):
        out_path = os.path.join(outdir, f"{cam_name}.npz")
        if os.path.exists(out_path) and not overwrite:
            continue

        frames = views[cam_idx]  # (T, H, W, 3), uint8 tensor from decord/cv2

        # Resize and pad to TARGET_RES × TARGET_RES (matching posetail's inference resolution)
        frames_np = frames.numpy() if isinstance(frames, torch.Tensor) else np.asarray(frames)
        frames_np, rp_scale, rp_pad_left, rp_pad_top = resize_and_pad(frames_np)
        video_size = np.array([TARGET_RES, TARGET_RES], dtype=np.int32)

        frames_t = torch.from_numpy(frames_np)  # (T, TARGET_RES, TARGET_RES, 3)

        # Project 3D queries to this camera's 2D pixel space, then into TARGET_RES space
        queries_2d = project_to_camera(camera_group, query_3d, cam_idx)  # (N, 2) original pixels
        queries_proc = queries_2d.clone()
        queries_proc[:, 0] = queries_2d[:, 0] * rp_scale + rp_pad_left
        queries_proc[:, 1] = queries_2d[:, 1] * rp_scale + rp_pad_top

        tracks, visibility = run_cotracker_single_camera(
            model, frames_t, queries_proc, device,
            query_batch_size=query_batch_size,
        )
        # tracks: (T, N, 2) in TARGET_RES space — invert to original pixel space before saving
        tracks_orig = tracks.clone()
        tracks_orig[:, :, 0] = (tracks[:, :, 0] - rp_pad_left) / rp_scale
        tracks_orig[:, :, 1] = (tracks[:, :, 1] - rp_pad_top)  / rp_scale

        tqdm.write(f"      {cam_name}: tracks {tuple(tracks.shape)}")
        np.savez(
            out_path,
            tracks=tracks_orig.numpy(),      # (T, N, 2) float32, pixel (x, y) in original space
            visibility=visibility.numpy(),   # (T, N) bool
            queries_2d=queries_2d.numpy(),   # (N, 2) in original pixel space
            frame_numbers=frame_numbers,     # (T,) int64
            cam_name=cam_name,
            video_size=video_size,           # (W, H) after any resizing
            valid_mask=valid_mask,           # (S*K,) bool
            start_frame=start_frame,
        )
        print(f"      {cam_name}: tracks {tuple(tracks.shape)}")

    torch.cuda.empty_cache()


# ── main loop ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True,
                   help="Root directory containing all datasets")
    p.add_argument("--datasets", nargs="+", required=True,
                   help="One or more dataset names (sub-folders of dataset-root)")
    p.add_argument("--output-root", required=True,
                   help="Root directory for outputs; mirrors dataset/split/session/trial structure")
    p.add_argument("--split", default="test",
                   help="Which split to process (default: test)")
    p.add_argument("--checkpoint", required=True,
                   help="Path to scaled_offline.pth")
    p.add_argument("--device", default="cuda",
                   help="Torch device (default: cuda)")
    p.add_argument("--window-len", type=int, default=60,
                   help="CoTracker offline window length (default: 60)")
    p.add_argument("--start-frame", type=int, default=0,
                   help="Frame index to begin tracking from (default: 0)")
    p.add_argument("--n-frames", type=int, default=None,
                   help="Max frames per trial; omit for full trial")
    p.add_argument("--max-res", type=int, default=None,
                   help="Downsample long edge to this many pixels before "
                        "passing to CoTracker (reduces VRAM usage)")
    p.add_argument("--query-batch-size", type=int, default=None,
                   help="Split queries into batches of this size to reduce VRAM usage. "
                        "Each batch runs independently (attention is local to the batch), "
                        "so results differ slightly from full-N inference.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run trials that already have cotracker/ outputs")
    p.add_argument("--pose-file", type=str, default=None,
                   help="Override pose3d.npz with this file for query points (e.g. pose3d_100.npz)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading CoTracker3 from {args.checkpoint}")
    model = CoTrackerPredictor(
        checkpoint=args.checkpoint,
        offline=True,
        window_len=args.window_len,
    ).to(device)
    model.eval()

    for dataset_name in args.datasets:
        split_path = os.path.join(args.dataset_root, dataset_name, args.split)
        sessions = sorted(get_dirs(split_path))

        # Collect all trials upfront so tqdm can show a total
        all_trials = [
            (session, trial,
             os.path.join(split_path, session, trial))
            for session in sessions
            for trial in sorted(get_dirs(os.path.join(split_path, session)))
        ]

        for session, trial, trial_path in tqdm(
            all_trials, desc=dataset_name, unit="trial"
        ):
            tqdm.write(f"  {session}/{trial}")
            outdir = os.path.join(
                args.output_root, dataset_name, args.split, session, trial, "cotracker"
            )
            process_trial(
                trial_path=trial_path,
                outdir=outdir,
                model=model,
                device=device,
                start_frame=args.start_frame,
                n_frames=args.n_frames,
                max_res=args.max_res,
                query_batch_size=args.query_batch_size,
                overwrite=args.overwrite,
                pose_file=args.pose_file,
            )

    print("Done.")


if __name__ == "__main__":
    main()