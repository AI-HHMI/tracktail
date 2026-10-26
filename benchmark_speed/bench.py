"""
Speed / GPU-memory benchmark of Tracktail and the baselines (CoTracker3 + triangulation, MVTracker with GT or
DUSt3R depth, ARROW) on a Point Odyssey test trial, as a function of the number of views and of query points.

One process runs one method (each baseline lives in its own Python environment; see run_all.sh). For every
(views, keypoints) configuration it runs the method's full inference pipeline on the same clip:

    decoded frames (uint8, native 960x540, already in CPU memory) + calibration + 3D queries
        -> method-specific preprocessing (resize / crop / pad, host->device copies)
        -> model(s)                      (+ triangulation for CoTracker, + DUSt3R depth for mvtracker-dust3r)
        -> 3D tracks for every frame, back in CPU memory

Timed: everything above. Not timed: reading/decoding the JPEGs (and the GT depth files) from disk, model loading.
Warmup: `--warmup` untimed runs of the same configuration first (CUDA context, cuDNN autotuning, lazy kernel
compilation, allocator growth), then `--repeats` timed runs; we report the median (and all times).
fps = n_frames / median wall time (torch.cuda.synchronize() before the clock is read).
GPU memory: torch.cuda.max_memory_allocated / max_memory_reserved over the timed runs, after empty_cache and
reset_peak_memory_stats (so each configuration starts from the model weights only), plus the model weights alone.

mvtracker-dust3r: DUSt3R (pairwise inference on the complete graph of the V views + 300 iterations of global
alignment with the known cameras, as mvtracker_baseline/precompute_dust3r_depths.py) runs per frame. Its cost does
not depend on the number of queries, so it is measured once per view count (`--dust3r-warmup` untimed frames, then
every one of the first `--dust3r-frames` frames timed) and the resulting depths feed the MVTracker stage, which is
timed per (views, keypoints) as above. Overall time = DUSt3R per-frame time * n_frames + MVTracker stage time;
peak memory = max of the two stages.

Queries: points of pose3d_full.npz visible at frame 0 in one of the first two cameras of the view order (so the
same points are valid for every view count), sampled with a fixed seed, nested (10 c 100 c 1000), all queried at
frame 0.

Usage (see run_all.sh for the environments):
    python bench.py --method tracktail --views 2 4 6 8 10 --kpts 10 100 1000 --out results/tracktail.jsonl

At 10 views, Point Odyssey's view2 is duplicated in the final input slot; this is not an independent camera.
"""

import argparse
import gc
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_TRIAL = ("/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6/"
                 "point-odyssey-human/test/hum0/trial")
# Camera slots in order: 2 views = first 2; the canonical PO views (2,4,5,7) are the first 4.
# Point Odyssey has only 9 cameras. The requested 10-view condition appends a duplicate of view2 in slot 10;
# it tests 10 model input slots, not 10 independent camera viewpoints.
VIEW_ORDER = ["view2", "view5", "view4", "view7", "view0", "view8", "view1", "view3", "view6", "view2"]

TRACKTAIL_RUN = "/groups/karashchuk/home/ruppk2/results/posetail-pretraining-v6/wandb/run-20260912_115732-vzkjqrys"
BUILDS = "/groups/karashchuk/home/karashchukl/builds"   # shared copies (the /home/<user>@hhmi.org dirs are workstation-local)
COTRACKER_CKPT = "/groups/karashchuk/home/ruppk2/software/cotracker-weights/scaled_offline.pth"
MVTRACKER_REPO = f"{BUILDS}/mvtracker"
MVTRACKER_CKPT = "/groups/karashchuk/karashchuklab/mvtracker-datasets/checkpoints/mvtracker_200000_june2025.pth"
DUST3R_CKPT = "/groups/karashchuk/karashchuklab/mvtracker-datasets/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

TARGET_RES = 256   # resize-and-pad resolution of the CoTracker / MVTracker pipelines (matches tracktail)


def natural_key(name):
    import re
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gb(x):
    return round(x / 2**30, 3)


# ── data ──────────────────────────────────────────────────────────────────────

class Clip:
    """Frames (V, T, H, W, 3) uint8 of the cameras in VIEW_ORDER[:max_views], calibration, and the query pool."""

    def __init__(self, trial, n_frames, max_views, max_kpts, seed=0):
        self.trial = trial
        with open(os.path.join(trial, "metadata.yaml")) as f:
            meta = yaml.safe_load(f)
        names = list(meta["intrinsic_matrices"].keys())
        self.names_all = sorted(names, key=int) if all(n.isdigit() for n in names) else sorted(names)
        self.cams = VIEW_ORDER[:max_views]
        self.cam_idx_all = [self.names_all.index(c) for c in self.cams]   # index into metadata / vis order
        self.K = np.stack([np.asarray(meta["intrinsic_matrices"][c], np.float64) for c in self.cams])
        self.E = np.stack([np.asarray(meta["extrinsic_matrices"][c], np.float64) for c in self.cams])
        self.dist = np.stack([np.asarray(meta["distortion_matrices"][c], np.float64) for c in self.cams])
        self.T = n_frames

        frames = []
        for c in self.cams:
            d = os.path.join(trial, "img", c)
            files = sorted(os.listdir(d), key=natural_key)[:n_frames]
            frames.append(np.stack([np.asarray(Image.open(os.path.join(d, f)).convert("RGB")) for f in files]))
        self.frames = np.stack(frames)                     # (V, T, H, W, 3)
        assert self.frames.shape[1] == n_frames, self.frames.shape
        self.H, self.W = self.frames.shape[2:4]

        pose = np.load(os.path.join(trial, "pose3d_full.npz"))
        xyz0 = pose["pose"][0, 0]                          # (K, 3) at frame 0
        vis0 = pose["vis"][0, 0][:, self.cam_idx_all]      # (K, V) visibility at frame 0, cameras in VIEW_ORDER
        ok = np.isfinite(xyz0).all(-1) & vis0[:, :2].any(-1)
        # also require the projection to be inside both of the first two images when visible there
        idx = np.nonzero(ok)[0]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(idx)[:max_kpts]
        assert len(idx) == max_kpts, f"only {len(idx)} valid query points"
        self.q3d = xyz0[idx].astype(np.float32)            # (N, 3), nested: the first n are the n-point set
        self.vis0 = vis0[idx]                              # (N, V)
        self.kpt_idx = idx
        # GT trajectories of the query points (only for a sanity check of the outputs, never given to a method)
        self.gt_traj = pose["pose"][0, :n_frames][:, idx].astype(np.float32)                   # (T, Nmax, 3)
        self.gt_vis = pose["vis"][0, :n_frames][:, idx][..., self.cam_idx_all]                  # (T, Nmax, V)

    def subset(self, V, N):
        return dict(frames=self.frames[:V], K=self.K[:V], E=self.E[:V], dist=self.dist[:V],
                    cams=self.cams[:V], q3d=self.q3d[:N], vis0=self.vis0[:N, :V])

    def gt_depths(self, max_views):
        """GT depth (T, V, H, W) float32 of the cameras in VIEW_ORDER[:max_views] (Point Odyssey gt-depths/)."""
        d = os.path.join(self.trial, "gt-depths")
        files = sorted(os.listdir(d), key=natural_key)[:self.T]
        idx = self.cam_idx_all[:max_views]
        return np.stack([np.load(os.path.join(d, f))["depths"][idx] for f in files]).astype(np.float32)


def project(X, K, E):
    Xc = X @ E[:3, :3].T + E[:3, 3]
    uv = Xc @ K.T
    return uv[:, :2] / uv[:, 2:3], Xc[:, 2]


def resize_pad_params(H, W, target=TARGET_RES):
    s = target / max(H, W)
    nh, nw = int(round(H * s)), int(round(W * s))
    return s, nh, nw, (target - nh) // 2, (target - nw) // 2


# ── methods ───────────────────────────────────────────────────────────────────

class Tracktail:
    """scripts/eval_testset_metrics.py settings: run_tracker_encoder_on_videos, n_overlap=8, motion margin,
    max_kpts=2500, GT occlusion at the query frame. The video readers are replaced by in-memory arrays."""

    def __init__(self, args, device):
        sys.path.insert(0, str(REPO))
        from tracktail.inference import inference_utils as iu
        self.iu = iu
        self.model, *_ = iu.load_model_from_base_folder(args.tracktail_run, device=device)
        self.model.eval()
        self.device = device
        self.cg_all = iu.load_camera_group_from_metadata(os.path.join(args.trial, "metadata.yaml"), device="cpu")
        # in-memory readers: numpy (T, H, W, 3) arrays support len() and slicing like the video readers
        iu.build_video_readers = lambda arrays: (list(arrays), [len(a) for a in arrays])
        self.n_overlap = args.tracktail_overlap

    def set_clip(self, clip):
        self.cg = [self.cg_all[i] for i in clip.cam_idx_all]

    def run(self, d):
        V = len(d["cams"])
        occ = torch.as_tensor(d["vis0"].astype(np.int64))
        res = self.iu.run_tracker_encoder_on_videos(
            model=self.model, video_paths=list(d["frames"]), camera_group=self.cg[:V],
            query_points_3d=torch.from_numpy(d["q3d"]), start_frame=0, n_frames=d["frames"].shape[1],
            n_overlap=self.n_overlap, max_kpts=2500, device=self.device, occlusion_gt=occ,
            query_times=None, motion_margin=True)
        return res["coords_pred"]


class CoTracker:
    """cotracker_baseline/: CoTracker3 offline (window 60) on each view separately (all queries projected into
    every view, 256x256 resize-and-pad input), then aniposelib triangulation of the visible 2D tracks (>= 2 views)."""

    def __init__(self, args, device):
        from cotracker.predictor import CoTrackerPredictor
        self.model = CoTrackerPredictor(checkpoint=args.cotracker_ckpt, offline=True, window_len=60).to(device)
        self.model.eval()
        self.device = device

    def set_clip(self, clip):
        from aniposelib.cameras import Camera, CameraGroup
        import cv2
        cams = []
        for c, K, E, dist in zip(clip.cams, clip.K, clip.E, clip.dist):
            rvec = cv2.Rodrigues(E[:3, :3])[0].ravel()
            cam = Camera(matrix=K, dist=dist, rvec=rvec, tvec=E[:3, 3], name=c)
            cam.set_size((clip.W, clip.H))
            cams.append(cam)
        self.cams = cams

    def run(self, d):
        import cv2
        from aniposelib.cameras import CameraGroup
        frames, V = d["frames"], len(d["cams"])
        T, H, W = frames.shape[1:4]
        s, nh, nw, pt, pl = resize_pad_params(H, W)
        N = len(d["q3d"])
        tracks_all, vis_all = [], []
        for v in range(V):
            padded = np.zeros((T, TARGET_RES, TARGET_RES, 3), np.uint8)
            padded[:, pt:pt + nh, pl:pl + nw] = np.stack([cv2.resize(f, (nw, nh)) for f in frames[v]])
            video = torch.from_numpy(padded).to(self.device).float().permute(0, 3, 1, 2)[None]
            uv, _ = project(d["q3d"].astype(np.float64), d["K"][v], d["E"][v])
            q = np.concatenate([np.zeros((N, 1)), uv * s + [pl, pt]], 1).astype(np.float32)
            with torch.no_grad():
                tracks, vis = self.model(video, queries=torch.from_numpy(q)[None].to(self.device))
            tr = tracks[0].cpu().numpy()
            tr[..., 0] = (tr[..., 0] - pl) / s
            tr[..., 1] = (tr[..., 1] - pt) / s
            tracks_all.append(tr)
            vis_all.append(vis[0].cpu().numpy().astype(bool))
        # triangulation (cotracker_triangulate.py)
        tracks = np.stack(tracks_all).astype(np.float32)
        vis = np.stack(vis_all)
        tracks[~vis] = np.nan
        cg = CameraGroup(self.cams[:V])
        p3d = cg.triangulate(tracks.reshape(V, T * N, 2), undistort=True).reshape(T, N, 3)
        p3d[vis.sum(0) < 2] = np.nan
        return p3d


class MVTracker:
    """mvtracker_baseline/mvtracker_inference.py --official-settings (Point Odyssey: visibility threshold 0.01,
    support grid 4, local grid 18, 4 iterations, no scene normalization), 256x256 resize-and-pad of frames and
    depths (the predictor resizes to 384x512), all queries in one call. depth = 'gt' or 'dust3r'."""

    def __init__(self, args, device, depth):
        sys.path.insert(0, MVTRACKER_REPO)
        kwargs = dict(visibility_threshold=0.01, grid_size=4, local_grid_size=18, n_iters=4)
        self.model = torch.hub.load(MVTRACKER_REPO, "mvtracker_predictor", source="local", pretrained=False,
                                    device=str(device), predictor_kwargs=kwargs)
        ckpt = torch.load(args.mvtracker_ckpt, map_location="cpu")
        self.model.model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        self.model.eval()
        self.device, self.depth = device, depth
        self.args = args
        if depth == "dust3r":
            from dust3r.model import AsymmetricCroCo3DStereo
            self.dust3r = AsymmetricCroCo3DStereo.from_pretrained(args.dust3r_ckpt).to(device).eval()
        self.dust3r_cache = {}     # V -> (depths (T, V, H, W) at native res, stage stats)

    def set_clip(self, clip):
        self.clip = clip
        if self.depth == "gt":
            self.gt = clip.gt_depths(len(clip.cams))      # (T, Vmax, H, W), loaded once, not timed

    # DUSt3R stage (precompute_dust3r_depths.py, in memory) ----------------------------------------------------
    def _dust3r_images(self, frames_t, size=512, patch=16):
        import torchvision.transforms as tvf
        norm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        imgs = []
        for i, f in enumerate(frames_t):
            img = Image.fromarray(f)
            S = max(img.size)
            img = img.resize(tuple(int(round(x * size / S)) for x in img.size),
                             Image.LANCZOS if S > size else Image.BICUBIC)
            W, H = img.size
            cx, cy = W // 2, H // 2
            hw, hh = ((2 * cx) // patch) * patch / 2, ((2 * cy) // patch) * patch / 2
            img = img.crop((cx - hw, cy - hh, cx + hw, cy + hh))
            imgs.append(dict(img=norm(img)[None], true_shape=np.int32([img.size[::-1]]), idx=i, instance=str(i)))
        return imgs

    def _dust3r_frame(self, frames_t, K, E, niter=300):
        import warnings
        from dust3r.cloud_opt import global_aligner
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.utils.device import to_numpy
        H0, W0 = frames_t.shape[1:3]
        images = self._dust3r_images(frames_t)
        th, tw = images[0]["true_shape"][0]
        rescale = tw / W0 if tw == 512 else th / H0
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        out = inference(pairs, self.dust3r, self.device, batch_size=1, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scene = global_aligner(out, device=self.device, verbose=False)
        focals = torch.tensor((K[:, 0, 0] + K[:, 1, 1]) / 2, dtype=torch.float32)
        pp = torch.tensor(K[:, :2, 2], dtype=torch.float32)
        scene.preset_focal(focals * rescale)
        scene.im_pp.requires_grad_(True)
        scene.preset_principal_point(pp * rescale)
        scene.preset_pose(torch.from_numpy(E).float().inverse())
        try:
            scene.compute_global_alignment(init="mst", niter=niter, schedule="linear", lr=0.01)
        except Exception:
            scene.compute_global_alignment(init="known_poses", niter=niter, schedule="linear", lr=0.01)
        scene = scene.clean_pointcloud()
        return np.stack(to_numpy(scene.get_depthmaps()))           # (V, h, w)

    def dust3r_stage(self, d):
        """DUSt3R depths for the V views of the clip (cached per V), with per-frame timing and peak memory."""
        import cv2
        V = len(d["cams"])
        if V in self.dust3r_cache:
            return self.dust3r_cache[V]
        frames = d["frames"]
        T = frames.shape[1]
        a = self.args
        for t in range(a.dust3r_warmup):
            self._dust3r_frame(frames[:, t], d["K"], d["E"])
        sync(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        n = min(T, a.dust3r_frames or T)
        depths, times = [], []
        for t in range(n):
            sync(); t0 = time.perf_counter()
            dep = self._dust3r_frame(frames[:, t], d["K"], d["E"])
            # back to native resolution, as load_precomputed_depths() does (part of the depth pipeline)
            dep = np.stack([cv2.resize(x, (self.clip.W, self.clip.H)) for x in dep])
            sync(); times.append(time.perf_counter() - t0)
            depths.append(dep)
        depths = np.stack([depths[t % n] for t in range(T)])        # frames beyond n reuse the measured ones
        stats = dict(dust3r_frame_times=times, dust3r_frames_timed=n,
                     dust3r_sec_per_frame=float(np.median(times)),
                     dust3r_fps=1.0 / float(np.median(times)),
                     dust3r_peak_alloc_gb=gb(torch.cuda.max_memory_allocated()),
                     dust3r_peak_reserved_gb=gb(torch.cuda.max_memory_reserved()))
        self.dust3r_cache[V] = (depths.astype(np.float32), stats)
        return self.dust3r_cache[V]

    # MVTracker stage ------------------------------------------------------------------------------------------
    def run(self, d, depths=None):
        import cv2
        frames, V = d["frames"], len(d["cams"])
        T, H, W = frames.shape[1:4]
        if depths is None:
            depths = self.gt[:, :V]
        s, nh, nw, pt, pl = resize_pad_params(H, W)
        rgb = np.zeros((V, T, TARGET_RES, TARGET_RES, 3), np.uint8)
        dep = np.zeros((V, T, TARGET_RES, TARGET_RES), np.float32)
        for t in range(T):
            for c in range(V):
                rgb[c, t, pt:pt + nh, pl:pl + nw] = cv2.resize(frames[c, t], (nw, nh))
                dep[c, t, pt:pt + nh, pl:pl + nw] = cv2.resize(depths[t, c], (nw, nh), interpolation=cv2.INTER_LINEAR)
        K = d["K"].copy()
        K[:, :2] *= s
        K[:, 0, 2] += pl
        K[:, 1, 2] += pt
        dev = self.device
        rgbs = torch.from_numpy(rgb).to(dev).permute(0, 1, 4, 2, 3).float()[None]            # (1, V, T, 3, H, W)
        dps = torch.from_numpy(dep).to(dev)[None, :, :, None]                               # (1, V, T, 1, H, W)
        intrs = torch.from_numpy(K).float().to(dev)[None, :, None].expand(1, V, T, 3, 3).contiguous()
        extrs = torch.from_numpy(d["E"][:, :3]).float().to(dev)[None, :, None].expand(1, V, T, 3, 4).contiguous()
        N = len(d["q3d"])
        q = torch.from_numpy(np.concatenate([np.zeros((N, 1), np.float32), d["q3d"]], 1)).to(dev)[None]
        with torch.no_grad():
            res = self.model(rgbs=rgbs, depths=dps, intrs=intrs, extrs=extrs, query_points_3d=q)
        return res["traj_e"][0].cpu().numpy(), res["vis_e"][0].cpu().numpy()


class Arrow:
    """arrow_baseline/arrow_inference.py: RGB only, each query a 2D pixel in one view that sees it (seeded choice),
    windows of 24 timesteps x V views (max side 504), queries outside a window added as anchor images, decode
    every query at every image of the window, scale fit per encoding (gt-tracks, paper Eq. 11)."""

    def __init__(self, args, device):
        from arrow import ARROW
        self.model = ARROW.from_pretrained(args.arrow_ckpt, device=str(device))
        self.model.eval()
        self.device, self.args = device, args

    def set_clip(self, clip):
        self.clip = clip
        # GT trajectories of the query points, for the gt-tracks scale fit (part of the ARROW protocol)
        self.gt_traj = clip.gt_traj.transpose(1, 0, 2).astype(np.float64)                   # (Nmax, T, 3)
        self.gt_vis = clip.gt_vis.transpose(1, 0, 2)                                        # (Nmax, T, V)

    def run(self, d):
        from arrow.data.transforms import prepare_images
        from arrow.types import Queries
        a, dev = self.args, self.device
        frames, V = d["frames"], len(d["cams"])
        T = frames.shape[1]
        N = len(d["q3d"])
        K, E = d["K"], d["E"]
        q3d = d["q3d"].astype(np.float64)
        rng = np.random.default_rng(0)
        src_view, src_xy = np.empty(N, np.int64), np.empty((N, 2))
        for n in range(N):
            cand = np.nonzero(d["vis0"][n])[0]
            cand = cand if len(cand) else np.arange(V)
            uvz = [project(q3d[n:n + 1], K[c], E[c]) for c in cand]
            front = [c for c, (_, z) in zip(cand, uvz) if z[0] > 0]
            c = rng.choice(front if front else cand)
            src_view[n], src_xy[n] = c, project(q3d[n:n + 1], K[c], E[c])[0][0]

        cache = {}

        def get(t, c):
            if (t, c) not in cache:
                info = {}
                ten = prepare_images([Image.fromarray(frames[c, t])], [info], max_side=a.arrow_max_side)[0]
                cache[(t, c)] = (ten, np.asarray(info["original_to_input"], np.float64))
            return cache[(t, c)]

        gt_cam0 = self.gt_traj[:N] @ E[0][:3, :3].T + E[0][:3, 3]
        vis_any = self.gt_vis[:N, :, :V].any(-1)
        pred = np.full((N, T, 3), np.nan)
        qt = np.zeros(N, np.int64)
        n_win = max(1, round(T / a.arrow_window))
        bounds = np.linspace(0, T, n_win + 1).round().astype(int)
        for w0, w1 in zip(bounds[:-1], bounds[1:]):
            win_t = list(range(w0, w1))
            Tw = len(win_t)
            inside = (qt >= w0) & (qt < w1)
            outside = sorted(set((int(qt[n]), int(src_view[n])) for n in np.nonzero(~inside)[0]))
            chunks = [outside[i:i + a.arrow_max_anchors] for i in range(0, len(outside), a.arrow_max_anchors)] or [[]]
            for ci, anchors in enumerate(chunks):
                apos = {k: Tw * V + i for i, k in enumerate(anchors)}
                sel = np.asarray([n for n in range(N) if (ci == 0 and inside[n]) or
                                  (int(qt[n]), int(src_view[n])) in apos])
                if len(sel) == 0:
                    continue
                keys = [(t, v) for t in win_t for v in range(V)] + list(anchors)
                loaded = [get(t, v) for t, v in keys]
                imgs = torch.stack([x[0] for x in loaded])
                sf, xy = [], []
                for n in sel:
                    k = (int(qt[n]), int(src_view[n]))
                    f = (k[0] - w0) * V + k[1] if inside[n] else apos[k]
                    A = loaded[f][1]
                    xy.append(A[:2, :2] @ src_xy[n] + A[:2, 2])
                    sf.append(f)
                Hh, Ww = imgs.shape[-2:]
                xy = np.clip(np.asarray(xy), -0.5, [Ww - 0.5, Hh - 0.5])
                F_win = Tw * V
                with torch.inference_mode():
                    enc = self.model.encode(imgs[None].to(dev), anchor_frame=0)
                    xy_t = torch.as_tensor(xy, dtype=torch.float32, device=dev)
                    Q = Queries(xy_t.repeat_interleave(F_win, 0),
                                torch.as_tensor(np.asarray(sf), device=dev).repeat_interleave(F_win),
                                torch.arange(F_win, device=dev).repeat(len(sel)), 0)
                    p = self.model.decode(enc, Q, decode_chunk_size=a.arrow_decode_chunk)
                pts = p.points[0].float().cpu().numpy().reshape(len(sel), Tw, V, 3)[:, :, 0]
                _ = p.visibility_logits[0].float().cpu().numpy()
                del enc, p
                g, m = gt_cam0[sel][:, win_t], vis_any[sel][:, win_t]
                ok = m & np.isfinite(pts).all(-1) & np.isfinite(g).all(-1)
                den = np.median(np.linalg.norm(pts[ok], axis=-1)) if ok.any() else 0
                sc = float(np.median(np.linalg.norm(g[ok], axis=-1)) / den) if den > 0 else 1.0
                pred[sel[:, None], np.asarray(win_t)[None]] = pts * sc
        E0inv = np.linalg.inv(E[0])
        return pred @ E0inv[:3, :3].T + E0inv[:3, 3]


# ── benchmark loop ────────────────────────────────────────────────────────────

def build(method, args, device):
    if method == "tracktail":
        return Tracktail(args, device)
    if method == "cotracker":
        return CoTracker(args, device)
    if method == "mvtracker-gt":
        return MVTracker(args, device, "gt")
    if method == "mvtracker-dust3r":
        return MVTracker(args, device, "dust3r")
    if method == "arrow":
        return Arrow(args, device)
    raise ValueError(method)


def to_tn3(method, out):
    """Prediction (T, N, 3) numpy of any method's run() output."""
    if isinstance(out, tuple):
        out = out[0]
    if torch.is_tensor(out):
        out = out.float().cpu().numpy()
    out = np.asarray(out, dtype=np.float64)
    if method == "tracktail":
        out = out[0]
    if method == "arrow":
        out = out.transpose(1, 0, 2)
    return out


def sanity_error(clip, V, pred):
    """Median 3D error (scene units) over frames where the point is visible in one of the V views."""
    gt = clip.gt_traj[:, :pred.shape[1]]
    m = clip.gt_vis[:, :pred.shape[1], :V].any(-1) & np.isfinite(gt).all(-1)
    err = np.linalg.norm(pred - gt, axis=-1)[m]
    return float(np.nanmedian(err)) if err.size else None, float(np.isfinite(err).mean()) if err.size else None


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True,
                   choices=["tracktail", "cotracker", "mvtracker-gt", "mvtracker-dust3r", "arrow"])
    p.add_argument("--views", type=int, nargs="+", default=[2, 4, 6, 8],
                   help="view-slot counts; at 10, slot 10 duplicates view2 because Point Odyssey has 9 cameras")
    p.add_argument("--kpts", type=int, nargs="+", default=[10, 100, 1000])
    p.add_argument("--n-frames", type=int, default=96, help="clip length (frames of every view)")
    p.add_argument("--warmup", type=int, default=1, help="untimed runs per configuration")
    p.add_argument("--repeats", type=int, default=3, help="timed runs per configuration")
    p.add_argument("--trial", default=DEFAULT_TRIAL)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True, help="jsonl file (appended)")
    p.add_argument("--tracktail-run", default=TRACKTAIL_RUN)
    p.add_argument("--tracktail-overlap", type=int, default=8)
    p.add_argument("--cotracker-ckpt", default=COTRACKER_CKPT)
    p.add_argument("--mvtracker-ckpt", default=MVTRACKER_CKPT)
    p.add_argument("--dust3r-ckpt", default=DUST3R_CKPT)
    p.add_argument("--dust3r-warmup", type=int, default=2, help="untimed DUSt3R frames per view count")
    p.add_argument("--dust3r-frames", type=int, default=None, help="timed DUSt3R frames per view count (default all)")
    p.add_argument("--arrow-ckpt", default=f"{BUILDS}/arrow-weights", help="ifradlin/arrow snapshot (or hub id)")
    p.add_argument("--arrow-window", type=int, default=24)
    p.add_argument("--arrow-max-anchors", type=int, default=24)
    p.add_argument("--arrow-max-side", type=int, default=504)
    p.add_argument("--arrow-decode-chunk", type=int, default=100_000)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[{args.method}] loading clip {args.trial} ({args.n_frames} frames, {max(args.views)} views)", flush=True)
    clip = Clip(args.trial, args.n_frames, max(args.views), max(args.kpts))
    print(f"[{args.method}] loading model", flush=True)
    m = build(args.method, args, device)
    m.set_clip(clip)
    sync(); gc.collect(); torch.cuda.empty_cache()
    weights_gb = gb(torch.cuda.memory_allocated())
    env = dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(device), torch=torch.__version__,
               python=platform.python_version(), lsf_job=os.environ.get("LSB_JOBID"))
    print(f"[{args.method}] env {env}, weights {weights_gb} GB", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for V in args.views:
        d3 = None
        if args.method == "mvtracker-dust3r":
            try:
                depths, d3 = m.dust3r_stage(clip.subset(V, 1))
            except RuntimeError as e:
                if not is_oom(e):
                    raise
                d3 = dict(dust3r_status="oom")
                depths = None
        for N in args.kpts:
            d = clip.subset(V, N)
            rec = dict(method=args.method, views=V, kpts=N, n_frames=clip.T, warmup=args.warmup,
                       repeats=args.repeats, trial=args.trial, cams=d["cams"],
                       view_note="slot 10 duplicates view2" if V == 10 else None,
                       weights_gb=weights_gb, **env)
            run = (lambda: m.run(d, depths=depths)) if args.method == "mvtracker-dust3r" else (lambda: m.run(d))
            try:
                if d3 is not None and d3.get("dust3r_status") == "oom":
                    raise torch.cuda.OutOfMemoryError("DUSt3R stage out of memory")
                for _ in range(args.warmup):
                    run()
                sync(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
                times = []
                for _ in range(args.repeats):
                    sync(); t0 = time.perf_counter()
                    out = run()
                    sync(); times.append(time.perf_counter() - t0)
                med_err, frac_finite = sanity_error(clip, V, to_tn3(args.method, out))
                del out
                rec.update(sanity_median_err=med_err, sanity_frac_finite=frac_finite)
                med = float(np.median(times))
                rec.update(status="ok", times=times, sec=med, fps=clip.T / med,
                           peak_alloc_gb=gb(torch.cuda.max_memory_allocated(device)),
                           peak_reserved_gb=gb(torch.cuda.max_memory_reserved(device)))
                if d3 is not None:
                    stage = dict(stage_sec=med, stage_fps=clip.T / med, stage_peak_alloc_gb=rec["peak_alloc_gb"],
                                 stage_peak_reserved_gb=rec["peak_reserved_gb"])
                    total = d3["dust3r_sec_per_frame"] * clip.T + med
                    rec.update(stage, **{k: v for k, v in d3.items()}, sec=total, fps=clip.T / total,
                               peak_alloc_gb=max(rec["peak_alloc_gb"], d3["dust3r_peak_alloc_gb"]),
                               peak_reserved_gb=max(rec["peak_reserved_gb"], d3["dust3r_peak_reserved_gb"]))
            except RuntimeError as e:
                if not is_oom(e):
                    raise
                rec.update(status="oom", error=str(e)[:300],
                           peak_alloc_gb=gb(torch.cuda.max_memory_allocated(device)),
                           peak_reserved_gb=gb(torch.cuda.max_memory_reserved(device)))
            gc.collect(); torch.cuda.empty_cache()
            print(json.dumps({k: v for k, v in rec.items() if k not in ("times", "dust3r_frame_times", "trial")}),
                  flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
