"""
Run MVTracker inference on MVTracker-native datasets (kubric-multiview,
dex-ycb-multiview, panoptic-multiview).

These datasets store images/cameras differently from the posetail format used by
mvtracker_inference.py. This script handles their native layouts while producing
the same output format: mvtracker_3d_{depth_subdir}.npz per trial.

Output structure (flat, no session level):
    <output_root>/<dataset>/test/<trial_id>/mvtracker_3d_{depth_subdir}.npz

Supported datasets and their depth subdirs:
    kubric-multiview    duster-views-0123456789   (10 views, digits concatenated)
    dex-ycb-multiview   duster-views-01234567     (8 views)
    panoptic-multiview  duster-views-27-16-14-8   (4 views, dash-separated)
                        duster-views-1-4-7-11
                        duster-views-1-7-14-20

Usage:
    cd ~
    export PYTHONPATH="/groups/karashchuk/home/ruppk2/github/mvtracker:/groups/karashchuk/home/ruppk2/github/dust3r:$PYTHONPATH"
    export TORCH_HOME=/groups/karashchuk/home/ruppk2/.cache/torch

    python mvtracker_inference_native.py \\
        --dataset-type kubric \\
        --data-root /groups/karashchuk/karashchuklab/mvtracker-datasets \\
        --output-root /groups/karashchuk/home/ruppk2/mvtracker-outputs \\
        --depth-subdir duster-views-0123456789 \\
        --checkpoint /groups/karashchuk/karashchuklab/mvtracker-datasets/checkpoints/mvtracker_200000_june2025.pth \\
        --device cuda:0 \\
        --max-queries 512
"""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import kornia.geometry
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mvtracker_native_selection as SEL   # official-style GT visibility + track/query selection


# ── depth subdir → view list parsing ─────────────────────────────────────────

def parse_views_from_depth_subdir(depth_subdir: str) -> list[int]:
    """
    Parse the list of view indices encoded in a depth subdir name.

    kubric/dex-ycb: 'duster-views-0123456789'  → [0,1,2,3,4,5,6,7,8,9]
    panoptic:       'duster-views-27-16-14-8'  → [27,16,14,8]
    """
    m = re.match(r"duster-views-(.+)", depth_subdir)
    if m is None:
        raise ValueError(f"Cannot parse views from depth_subdir: {depth_subdir!r}")
    s = m.group(1)
    if "-" in s:
        return list(map(int, s.split("-")))
    else:
        return list(map(int, s))  # each digit is a view index


# ── depth loading ─────────────────────────────────────────────────────────────

def load_precomputed_depths(trial_dir: Path, frame_numbers: np.ndarray,
                            H: int, W: int, depth_subdir: str) -> np.ndarray:
    depth_dir = trial_dir / depth_subdir
    depths_list = []
    for frame_idx in frame_numbers:
        npz_path = depth_dir / f"3d_model__{int(frame_idx):05d}__scene.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Precomputed depth not found: {npz_path}\n"
                f"Run the precompute script first."
            )
        data   = np.load(npz_path)
        depths = data["depths"].astype(np.float32)   # (C, H_d, W_d)
        if depths.shape[-2:] != (H, W):
            C = depths.shape[0]
            depths = np.stack([cv2.resize(depths[c], (W, H)) for c in range(C)])
        depths_list.append(depths)
    return np.stack(depths_list)   # (T, C, H, W)


def load_kubric_gt_depths(trial_dir: Path, views: list[int],
                          frame_numbers: np.ndarray) -> np.ndarray:
    """Return (T, C, H, W) float32 z-depth from view_{v}/depth_{t:05d}.tiff (converted from euclidean)."""
    rows = []
    # Kubric's depth tiffs hold the EUCLIDEAN distance from the camera centre. MVTracker's loader converts
    # them to z-depth (KubricMultiViewDataset.depth_from_euclidean_to_z) and marks values > 1000 invalid (0).
    cam_params = {}
    for v in views:
        with open(trial_dir / f"view_{v}" / "metadata.json") as f:
            _c = json.load(f)["camera"]
        cam_params[v] = (_c["sensor_width"], _c["focal_length"])

    def _euclid_to_z(d, sensor_width, focal_length):
        h, w = d.shape
        sensor_height = sensor_width / w * h
        px = (np.arange(-w / 2, w / 2, dtype=np.float32) + 0.5) / w * sensor_width
        py = (np.arange(-h / 2, h / 2, dtype=np.float32) + 0.5) / h * sensor_height
        px, py = np.meshgrid(px, py, indexing="xy")
        return d / np.sqrt(1 + (np.square(px) + np.square(py)) / focal_length ** 2)

    for t in frame_numbers:
        cams = []
        for v in views:
            path = trial_dir / f"view_{v}" / f"depth_{int(t):05d}.tiff"
            d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if d is None:
                raise FileNotFoundError(f"GT depth not found: {path}")
            d = _euclid_to_z(d.astype(np.float32), *cam_params[v])
            d[d > 1000] = 0.0
            cams.append(d.astype(np.float32))
        rows.append(np.stack(cams, axis=0))   # (C, H, W)
    return np.stack(rows, axis=0)              # (T, C, H, W)


def load_panoptic_gt_depths(trial_dir: Path, views: list[int],
                             frame_numbers: np.ndarray) -> np.ndarray:
    """Return (T, C, H, W) float32 from dynamic3dgs_depth/depths_{v:02d}.npy."""
    cams = []
    for v in views:
        path = trial_dir / "dynamic3dgs_depth" / f"depths_{v:02d}.npy"
        if not path.exists():
            raise FileNotFoundError(f"GT depth not found: {path}")
        d = np.load(path).astype(np.float32)   # (T_full, H, W)
        cams.append(d[frame_numbers])           # (T, H, W)
    return np.stack(cams, axis=1)               # (T, C, H, W)


# ── kubric loaders ────────────────────────────────────────────────────────────

def load_kubric_cameras(trial_dir: Path, views: list[int]):
    """
    Return pixel-space intrinsics (C,3,3) and w2c extrinsics (C,4,4).
    Applies the coordinate flip from KubricMultiViewDataset.getitem_raw_datapoint.
    """
    intrs_list, extrs_list = [], []
    for v in views:
        with open(trial_dir / f"view_{v}" / "metadata.json") as f:
            meta = json.load(f)
        cam = meta["camera"]
        W, H = meta["metadata"]["resolution"]
        K_norm = np.array(cam["K"], dtype=np.float64)
        K_px   = np.diag([W, H, 1.0]) @ K_norm @ np.diag([1.0, -1.0, -1.0])

        positions   = torch.tensor(cam["positions"],   dtype=torch.float64)
        quaternions = torch.tensor(cam["quaternions"], dtype=torch.float64)
        R           = kornia.geometry.quaternion_to_rotation_matrix(quaternions)

        c2w = torch.zeros(4, 4, dtype=torch.float64)
        c2w[:3, :3] = R[0]
        c2w[:3, 3]  = positions[0]
        c2w[3, 3]   = 1.0
        w2c_4x4  = c2w.inverse()
        E_3x4    = w2c_4x4[:3, :].numpy()
        E_3x4    = np.diag([1.0, -1.0, -1.0]) @ E_3x4
        E_4x4    = np.eye(4, dtype=np.float64)
        E_4x4[:3, :] = E_3x4

        intrs_list.append(K_px.astype(np.float32))
        extrs_list.append(E_4x4.astype(np.float32))
    return np.stack(intrs_list), np.stack(extrs_list)   # (C,3,3), (C,4,4)


def load_kubric_images(trial_dir: Path, views: list[int]) -> np.ndarray:
    """Return (T, C, H, W, 3) uint8, dropping alpha channel from RGBA PNGs."""
    frame_lists = []
    for v in views:
        paths = sorted((trial_dir / f"view_{v}").glob("rgba_*.png"))
        if not paths:
            raise FileNotFoundError(f"No rgba_*.png in {trial_dir}/view_{v}")
        frame_lists.append(paths)
    T = len(frame_lists[0])
    H, W = None, None
    frames = []
    for t in range(T):
        row = []
        for vi in range(len(views)):
            img = cv2.imread(str(frame_lists[vi][t]), cv2.IMREAD_UNCHANGED)
            if img.shape[2] == 4:
                img = img[:, :, :3]           # drop alpha
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if H is None:
                H, W = img.shape[:2]
            row.append(img)
        frames.append(np.stack(row, axis=0))  # (C, H, W, 3)
    return np.stack(frames, axis=0)            # (T, C, H, W, 3)


def load_kubric_queries(trial_dir: Path, max_queries: int | None = None,
                        query_seed: int = 0):
    """Query at frame 0 from tracks_3d.npz."""
    data      = np.load(trial_dir / "tracks_3d.npz")
    tracks    = data["tracks_3d"]   # (T, N, 3)
    query_3d  = tracks[0].astype(np.float32)   # (N, 3)
    valid     = np.all(np.isfinite(query_3d), axis=1)
    if max_queries is not None:
        vi = np.where(valid)[0]
        if len(vi) > max_queries:
            rng    = np.random.default_rng(query_seed)
            chosen = rng.choice(vi, max_queries, replace=False)
            chosen.sort()
            valid  = np.zeros(len(valid), dtype=bool)
            valid[chosen] = True
    return query_3d[valid], valid


# ── dex-ycb loaders ───────────────────────────────────────────────────────────

def load_dexycb_cameras(trial_dir: Path, views: list[int]):
    """Return pixel-space intrinsics (C,3,3) and w2c extrinsics (C,4,4)."""
    intrs_list, extrs_list = [], []
    for v in views:
        d = np.load(trial_dir / f"view_{v:02d}" / "intrinsics_extrinsics.npz")
        K4 = np.array(d["intrinsics"], dtype=np.float32)   # (4,4)
        E  = np.array(d["extrinsics"], dtype=np.float32)   # (4,4) w2c
        intrs_list.append(K4[:3, :3])
        extrs_list.append(E)
    return np.stack(intrs_list), np.stack(extrs_list)


def load_dexycb_images(trial_dir: Path, views: list[int]) -> np.ndarray:
    """Return (T, C, H, W, 3) uint8."""
    frame_lists = []
    for v in views:
        paths = sorted((trial_dir / f"view_{v:02d}" / "rgb").glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No png in {trial_dir}/view_{v:02d}/rgb")
        frame_lists.append(paths)
    T = len(frame_lists[0])
    frames = []
    for t in range(T):
        row = []
        for vi in range(len(views)):
            img = cv2.imread(str(frame_lists[vi][t]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            row.append(img)
        frames.append(np.stack(row, axis=0))
    return np.stack(frames, axis=0)


def load_dexycb_queries(trial_dir: Path, max_queries: int | None = None,
                        query_seed: int = 0):
    """Query at frame 0; use tracks_3d_visibilities for valid mask."""
    data     = np.load(trial_dir / "tracks_3d.npz", allow_pickle=True)
    tracks   = np.array(data["tracks_3d"], dtype=np.float32)     # (T, N, 3)
    vis      = np.array(data["tracks_3d_visibilities"])           # (T, N)
    query_3d = tracks[0]
    valid    = vis[0] & np.all(np.isfinite(query_3d), axis=1)
    if max_queries is not None:
        vi = np.where(valid)[0]
        if len(vi) > max_queries:
            rng = np.random.default_rng(query_seed)
            chosen = rng.choice(vi, max_queries, replace=False)
            chosen.sort()
            valid = np.zeros(len(valid), dtype=bool)
            valid[chosen] = True
    return query_3d[valid], valid


# ── panoptic loaders ──────────────────────────────────────────────────────────

def load_panoptic_cameras(trial_dir: Path, views: list[int]):
    """Select (C,3,3) intrinsics and (C,4,4) extrinsics from tapvid3d_annotations.npz."""
    ann      = np.load(trial_dir / "tapvid3d_annotations.npz")
    intrs_all = ann["intrinsics"]   # (31, 3, 3)
    extrs_all = ann["extrinsics"]   # (31, 4, 4) w2c
    intrs = np.stack([intrs_all[v] for v in views]).astype(np.float32)
    extrs = np.stack([extrs_all[v] for v in views]).astype(np.float32)
    return intrs, extrs


def load_panoptic_images(trial_dir: Path, views: list[int]) -> np.ndarray:
    """Return (T, C, H, W, 3) uint8."""
    frame_lists = []
    for v in views:
        paths = sorted((trial_dir / "ims" / str(v)).glob("*.jpg"))
        if not paths:
            raise FileNotFoundError(f"No jpg in {trial_dir}/ims/{v}")
        frame_lists.append(paths)
    T = len(frame_lists[0])
    frames = []
    for t in range(T):
        row = []
        for vi in range(len(views)):
            img = cv2.imread(str(frame_lists[vi][t]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            row.append(img)
        frames.append(np.stack(row, axis=0))
    return np.stack(frames, axis=0)


def load_panoptic_queries(trial_dir: Path, max_queries: int | None = None,
                          query_seed: int = 0):
    """
    Query at frame 0 using positions from trajectories[0].
    query_points_3d[:,0] gives the canonical query frame per point, but we
    simplify to frame 0 for all points for compatibility with our inference loop.
    """
    ann       = np.load(trial_dir / "tapvid3d_annotations.npz")
    trajs     = ann["trajectories"]              # (T, N, 3)
    query_3d  = trajs[0].astype(np.float32)      # (N, 3) at frame 0
    valid     = np.all(np.isfinite(query_3d), axis=1)
    if max_queries is not None:
        vi = np.where(valid)[0]
        if len(vi) > max_queries:
            rng = np.random.default_rng(query_seed)
            chosen = rng.choice(vi, max_queries, replace=False)
            chosen.sort()
            valid = np.zeros(len(valid), dtype=bool)
            valid[chosen] = True
    return query_3d[valid], valid


# ── per-trial inference ───────────────────────────────────────────────────────

POSETAIL_SUBDIR = {
    "kubric":   ("kubric-multiview", lambda name: f"test/{name}/trial"),
    "panoptic": ("cmupanoptic_3dgs", lambda name: f"test/{name}/trial"),
    "dex-ycb":  ("dex_ycb",         lambda name: f"test/{name}/trial"),
}

# How each dataset names its per-view image subdirectories
def _view_dir_name(dataset_type: str, v: int) -> str:
    if dataset_type == "dex-ycb":
        return f"view_{v:02d}"   # view_00, view_01, ...
    elif dataset_type == "panoptic":
        return str(v)            # 0, 1, 2, ...
    else:
        return f"view_{v}"       # view_0, view_1, ... (kubric)


def posetail_trial_paths(posetail_root: Path, dataset_type: str,
                         trial_name: str, views: list[int]):
    """Return (metadata_path, video_paths) for render_video.py compatibility."""
    ds_dir, trial_fn = POSETAIL_SUBDIR[dataset_type]
    trial_path = posetail_root / ds_dir / trial_fn(trial_name)
    metadata_path = str(trial_path / "metadata.yaml")
    img_root = trial_path / "img"
    video_paths = [str(img_root / _view_dir_name(dataset_type, v)) for v in views]
    return metadata_path, video_paths


def process_trial(
    dataset_type: str,
    trial_dir: Path,
    outdir: Path,
    views: list[int],
    mvtracker_model,
    device: torch.device,
    depth_subdir: str | None = None,
    depth_type: str = "duster",
    max_queries: int | None = None,
    query_seed: int | None = None,
    overwrite: bool = False,
    posetail_root: Path | None = None,
    trial_index: int = 0,
    use_cached_tracks: bool = True,
):
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"    views={views}  depth_type={depth_type}" +
          (f"  depth_subdir={depth_subdir}" if depth_type == "duster" else ""))
    if depth_type == "duster":
        depth_dir = trial_dir / depth_subdir
        if not depth_dir.is_dir():
            print(f"    [skip] depth dir not found: {depth_dir}")
            return

    out_path = outdir / "mvtracker_3d.npz"
    if out_path.exists() and not overwrite:
        print("    [skip] already done")
        return

    # -- load images and cameras ------------------------------------------------
    if dataset_type == "kubric":
        frames_all = load_kubric_images(trial_dir, views)
        intrs_np, extrs_np = load_kubric_cameras(trial_dir, views)
    elif dataset_type == "dex-ycb":
        frames_all = load_dexycb_images(trial_dir, views)
        intrs_np, extrs_np = load_dexycb_cameras(trial_dir, views)
    elif dataset_type == "panoptic":
        frames_all = load_panoptic_images(trial_dir, views)
        intrs_np, extrs_np = load_panoptic_cameras(trial_dir, views)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

    # Tracks and queries chosen with MVTracker's own policy (visibility from the selected views; each
    # query at the track's first visible frame, a quarter at a random visible frame for dex-ycb/kubric)
    sel = SEL.select_tracks(dataset_type, trial_dir, views, n_queries=max_queries,
                            seed=query_seed, trial_index=trial_index, use_cache=use_cached_tracks)
    print(f"    tracks/queries from: {sel['source']}"
          + (f" ({Path(sel['cache_file']).name})" if sel['source'] == 'cache' else " (re-derived with the official policy)"))
    query_3d, query_t, valid_mask = sel["query_xyz"], sel["query_t"], sel["valid_mask"]
    if len(query_3d) == 0:
        print("    [skip] no valid query points")
        return

    T, C, H, W, _ = frames_all.shape
    frame_numbers = np.arange(T, dtype=np.int64)

    # -- depths -----------------------------------------------------------------
    print(f"    loading depths ({T} frames × {C} cams, type={depth_type})…")
    if depth_type == "duster":
        depths_np = load_precomputed_depths(trial_dir, frame_numbers, H, W, depth_subdir)
        # Subdir may contain more views than we're using (e.g. N=1 piggybacks on a 2-view precompute).
        # Select only the channels corresponding to the active views.
        depth_views = parse_views_from_depth_subdir(depth_subdir)
        depth_indices = [depth_views.index(v) for v in views]
        depths_np = depths_np[:, depth_indices]
    elif depth_type == "gt":
        if dataset_type == "kubric":
            depths_np = load_kubric_gt_depths(trial_dir, views, frame_numbers)
        elif dataset_type == "panoptic":
            depths_np = load_panoptic_gt_depths(trial_dir, views, frame_numbers)
        else:
            raise ValueError(f"No GT depths available for dataset_type={dataset_type!r}")
    else:
        raise ValueError(f"Unknown depth_type: {depth_type!r}")

    if os.environ.get('MVT_NATIVE_RES') != '1':
        # -- resize + pad to 256×256 (matching posetail's inference resolution) -----
        TARGET_RES  = 256
        rp_scale    = TARGET_RES / max(H, W)
        new_H_rp    = int(round(H * rp_scale))
        new_W_rp    = int(round(W * rp_scale))
        pad_top_rp  = (TARGET_RES - new_H_rp) // 2
        pad_left_rp = (TARGET_RES - new_W_rp) // 2
        C_depth = depths_np.shape[1]
        frames_padded = np.zeros((T, C, TARGET_RES, TARGET_RES, 3), dtype=np.uint8)
        depths_padded = np.zeros((T, C_depth, TARGET_RES, TARGET_RES), dtype=np.float32)
        for t in range(T):
            for c in range(C):
                f_r = cv2.resize(frames_all[t, c], (new_W_rp, new_H_rp))
                frames_padded[t, c, pad_top_rp:pad_top_rp + new_H_rp, pad_left_rp:pad_left_rp + new_W_rp] = f_r
            for c in range(C_depth):
                d_r = cv2.resize(depths_np[t, c], (new_W_rp, new_H_rp), interpolation=cv2.INTER_LINEAR)
                depths_padded[t, c, pad_top_rp:pad_top_rp + new_H_rp, pad_left_rp:pad_left_rp + new_W_rp] = d_r
        frames_all = frames_padded
        depths_np  = depths_padded
        # Adjust intrinsics for resize+pad (intrs_np is (C,3,3) here, tiled to (T,C,3,3) below)
        intrs_np = intrs_np.copy()
        intrs_np[:, 0, :] *= rp_scale     # scale fx and cx
        intrs_np[:, 1, :] *= rp_scale     # scale fy and cy
        intrs_np[:, 0, 2] += pad_left_rp  # shift cx for left pad
        intrs_np[:, 1, 2] += pad_top_rp   # shift cy for top pad
        H, W = TARGET_RES, TARGET_RES

    # -- scene normalization (MVTracker's dataloaders apply a per-dataset similarity transform) --
    norm_info = None
    if os.environ.get("MVT_SCENE_NORM") == "1" and dataset_type in ("panoptic", "dex-ycb"):
        from scipy.spatial.transform import Rotation as _R
        from mvtracker.datasets.utils import transform_scene
        if dataset_type == "panoptic":
            _scale = 2.5
            _rot = _R.from_euler('x', -90, degrees=True).as_matrix()
            _tr = np.array([0.0, 0.0, 0.0])
        else:
            _scale = 6.0
            _rot = (_R.from_euler('z', -30, degrees=True).as_matrix()
                    @ _R.from_euler('y', 3, degrees=True).as_matrix()
                    @ _R.from_euler('x', 220, degrees=True).as_matrix())
            _tr = np.array([0.0, 0.0, 0.5])
        _Tn = depths_np.shape[0]
        _d = torch.from_numpy(depths_np.transpose(1, 0, 2, 3))[:, :, None].float()                 # [V,T,1,H,W]
        _e = torch.from_numpy(np.broadcast_to(extrs_np[:, None, :3, :], (extrs_np.shape[0], _Tn, 3, 4)).copy()).float()
        _q = torch.from_numpy(np.concatenate([query_t[:, None].astype(np.float64), query_3d], 1)).float()
        _d2, _e2, _q2, _, _ = transform_scene(_scale, torch.from_numpy(_rot).float(), torch.from_numpy(_tr).float(),
                                              _d, _e, _q, None, None)
        depths_np = _d2[:, :, 0].numpy().transpose(1, 0, 2, 3).copy()
        extrs_np = np.concatenate([_e2[:, 0].numpy(), np.broadcast_to(np.array([[[0, 0, 0, 1.0]]]), (extrs_np.shape[0], 1, 4))], axis=1).astype(np.float32)
        query_3d = _q2[:, 1:].numpy().astype(np.float32)
        norm_info = dict(norm_scale=np.array(_scale), norm_rot=_rot, norm_translation=_tr)
        print(f"    scene normalization applied: scale={_scale}")

    # -- build MVTracker tensors ------------------------------------------------
    # model convention: (batch, views=C, frames=T, ...)
    # our arrays are (T, C, ...) so we transpose T↔C everywhere

    # (T,C,H,W,3) → (C,T,3,H,W) → (1,C,T,3,H,W)
    # Model normalizes internally with /255.0; pass float values in [0, 255]
    rgbs_t = torch.from_numpy(
        frames_all.transpose(1, 0, 4, 2, 3).astype(np.float32)
    ).unsqueeze(0).to(device)

    # depths_np (T,C,H,W) → (C,T,H,W) → (1,C,T,1,H,W)
    depths_t = (torch.from_numpy(depths_np.transpose(1, 0, 2, 3))
                .unsqueeze(0).unsqueeze(3).to(device))

    # intrs: (C,3,3) → tile to (T,C,3,3) → (C,T,3,3) → (1,C,T,3,3)
    intrs_t_np = np.broadcast_to(intrs_np[np.newaxis], (T, C, 3, 3)).copy()
    intrs_t = torch.from_numpy(intrs_t_np.transpose(1, 0, 2, 3)).unsqueeze(0).to(device)

    # extrs: (C,4,4) → tile to (T,C,4,4) → (C,T,4,4) → drop last row → (1,C,T,3,4)
    extrs_t_np = np.broadcast_to(extrs_np[np.newaxis], (T, C, 4, 4)).copy()
    extrs_t = (torch.from_numpy(extrs_t_np.transpose(1, 0, 2, 3)[:, :, :3, :])
               .unsqueeze(0).to(device))

    # queries at each track's query frame → (1, N, 4) = [t_idx, x, y, z]
    t_idx   = query_t[:, None].astype(np.float32)   # per-track query frame
    query_4d = np.concatenate([t_idx, query_3d], axis=1)
    queries_t = torch.from_numpy(query_4d).unsqueeze(0).to(device)

    # -- run MVTracker ----------------------------------------------------------
    print(f"    running MVTracker (N={len(query_3d)})…")
    with torch.no_grad():
        results = mvtracker_model(
            rgbs=rgbs_t,
            depths=depths_t,
            intrs=intrs_t,
            extrs=extrs_t,
            query_points_3d=queries_t,
        )

    traj_e       = results["traj_e"].squeeze(0).cpu().numpy()        # (T, N, 3)
    traj_e_normalized = None
    if norm_info is not None:
        # model output lives in the normalized scene; map back to the dataset's raw world frame so
        # downstream metric/render scripts (which use raw GT) work. X' = R (s X) + t  =>  X = R^T (X' - t) / s
        traj_e_normalized = traj_e.copy()
        traj_e = ((traj_e - norm_info["norm_translation"]) @ norm_info["norm_rot"]) / float(norm_info["norm_scale"])
    vis_e        = results["vis_e"].squeeze(0).cpu().numpy()         # (T, N) bool
    vis_e_prob   = results["vis_e_as_prob"].squeeze(0).cpu().numpy() # (T, N) float prob
    visibility   = vis_e.astype(bool)

    # Convert probability → logits for render_video.py (which applies sigmoid)
    eps      = 1e-6
    vis_prob = np.clip(vis_e_prob, eps, 1 - eps).astype(np.float32)
    vis_logits = np.log(vis_prob / (1 - vis_prob))                   # (T, N)
    vis_pred_4d = vis_logits[np.newaxis, :, :, np.newaxis]           # (1, T, N, 1)

    cam_names = [f"view_{v}" for v in views]
    save_dict = dict(
        coords_3d    = traj_e.astype(np.float32),
        visibility   = visibility,
        frame_numbers= frame_numbers,
        valid_mask   = valid_mask,
        query_times  = query_t.astype(np.int64),
        cam_names    = np.array(cam_names, dtype=str),
        cam_indices  = np.array(views, dtype=np.int32),
        # render_video.py-compatible keys
        coords_pred  = traj_e[np.newaxis].astype(np.float32),  # (1, T, N, 3)
        vis_pred     = vis_pred_4d,                             # (1, T, N, 1) logits
        conf_pred    = vis_pred_4d[..., 0],                     # (1, T, N)   logits
    )
    if norm_info is not None:
        save_dict.update(norm_info)
        save_dict['coords_3d_normalized'] = traj_e_normalized.astype(np.float32)
    if posetail_root is not None:
        metadata_path, video_paths = posetail_trial_paths(
            posetail_root, dataset_type, trial_dir.name, views)
        save_dict["metadata_path"] = np.array(metadata_path, dtype=str)
        save_dict["video_paths"]   = np.array(video_paths,   dtype=str)
    np.savez(str(out_path), **save_dict)
    print(f"    saved: {tuple(traj_e.shape)}  cam_indices={views}")
    torch.cuda.empty_cache()


# ── dataset enumeration ───────────────────────────────────────────────────────

def find_trials_native(data_root: Path, dataset_type: str) -> list[Path]:
    """Return sorted list of trial directories for each dataset type."""
    if dataset_type == "kubric":
        base = data_root / "kubric-multiview" / "test"
    elif dataset_type == "dex-ycb":
        base = data_root / "dex-ycb-multiview"
    elif dataset_type == "panoptic":
        base = data_root / "panoptic-multiview"
    else:
        raise ValueError(dataset_type)
    return sorted([d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")])


def dataset_name(dataset_type: str) -> str:
    return {"kubric": "kubric-multiview", "dex-ycb": "dex-ycb-multiview",
            "panoptic": "panoptic-multiview"}[dataset_type]


# Total camera counts per dataset type (used for random subsampling)
TOTAL_VIEWS = {"kubric": 10, "dex-ycb": 8, "panoptic": 31}

# Camera subsets from MVTracker's main evaluation table (one combo per dataset).
# Ablation combos for panoptic ([27,16,14,8] and [1,4,7,11]) are not included here;
# use --views explicitly if you want to replicate those.
MVTRACKER_CANONICAL_VIEWS = {
    "kubric":   [0, 1, 2, 3],
    "dex-ycb":  [0, 1, 2, 3],
    "panoptic": [1, 7, 14, 20],
}


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-type", required=True,
                   choices=["kubric", "dex-ycb", "panoptic"],
                   help="Which dataset format to use")
    p.add_argument("--data-root", required=True,
                   help="Root containing kubric-multiview/, dex-ycb-multiview/, panoptic-multiview/")
    p.add_argument("--output-root", required=True,
                   help="Where to write outputs (same structure as Point Odyssey outputs)")
    p.add_argument("--views", type=int, nargs="+", default=None,
                   help="Explicit view indices; overrides --n-views and --view-seed")
    p.add_argument("--mvtracker-canonical", action="store_true",
                   help="Use MVTRACKER_CANONICAL_VIEWS for this dataset (main-table cameras)")
    p.add_argument("--n-views", type=int, default=None,
                   help="Use only N randomly-selected cameras; None = all")
    p.add_argument("--view-seed", type=int, default=0,
                   help="RNG seed for camera subsampling")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--depth-type", default="duster", choices=["duster", "gt"],
                   help="duster: precomputed DUSt3R depths; gt: dataset GT depths")
    p.add_argument("--max-queries", type=int, default=None,
                   help="number of tracks to query; default = the official count (kubric 512, dex-ycb 384, panoptic 384)")
    p.add_argument("--query-seed", type=int, default=None,
                   help="sampling seed; setting it re-samples the tracks instead of using the cached ones "
                        "(default: use the authors' cached tracks; otherwise the official seed 72, kubric 72 + dataset index)")
    p.add_argument("--no-cached-tracks", action="store_true",
                   help="do not read MVTracker's cached track files (<trial>/cache/cachedtracks--*.npz); re-derive the "
                        "tracks and queries with the official policy instead. Also used automatically when no cache "
                        "exists for the chosen cameras or when --max-queries differs from the cache size")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--depth-subdir", type=str, default=None,
                   help="Override the auto-computed DUSt3R depth subdir name")
    p.add_argument("--posetail-root", type=str, default=None,
                   help="Root of posetail-pretraining-v6 to save video_paths/metadata_path "
                        "for render_video.py compatibility")
    return p.parse_args()


def main():
    args        = parse_args()
    device      = torch.device(args.device)
    data_root   = Path(args.data_root)
    output_root = Path(args.output_root)

    total_views = TOTAL_VIEWS[args.dataset_type]
    if args.mvtracker_canonical:
        views = MVTRACKER_CANONICAL_VIEWS[args.dataset_type]
        print(f"MVTracker canonical views: {views}")
    elif args.views is not None:
        views = sorted(args.views)
        print(f"Explicit views: {views}")
    elif args.n_views is not None and total_views > args.n_views:
        rng   = np.random.default_rng(args.view_seed)
        views = sorted(rng.choice(total_views, args.n_views, replace=False).tolist())
        print(f"Subsampled {args.n_views}/{total_views} views: {views}")
    else:
        views = list(range(total_views))
        print(f"Using all {total_views} views")
    depth_type = args.depth_type
    if depth_type == "duster":
        if args.depth_subdir is not None:
            depth_subdir = args.depth_subdir
        else:
            depth_subdir = "duster-views-" + "-".join(str(v) for v in views)
        print(f"Depth subdir: {depth_subdir}")
    else:
        depth_subdir = None

    print(f"Loading MVTracker from {args.checkpoint}…")
    _PS = {"kubric": dict(visibility_threshold=0.5, grid_size=4, local_grid_size=18),
           "dex-ycb": dict(visibility_threshold=0.01, grid_size=4, local_grid_size=18),
           "panoptic": dict(visibility_threshold=0.01, grid_size=6, local_grid_size=18)}[args.dataset_type]
    _PS["n_iters"] = int(os.environ.get("MVT_ITERS", "4"))
    print("predictor settings:", _PS)
    mvtracker_model = torch.hub.load(
        "ethz-vlg/mvtracker", "mvtracker_predictor", pretrained=False, device=str(device),
        predictor_kwargs=_PS
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # Checkpoint keys have no "model." prefix but EvaluationPredictor stores
    # the inner MVTracker as self.model — load into the inner model directly.
    mvtracker_model.model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    mvtracker_model.eval()

    dname  = dataset_name(args.dataset_type)
    trials = find_trials_native(data_root, args.dataset_type)
    if args.dataset_type == "kubric":   # official kubric seed = 72 + index in numeric sequence order
        _order = {p.name: i for i, p in enumerate(sorted(trials, key=lambda p: int(p.name)))}
    else:
        _order = {}
    print(f"\nFound {len(trials)} trial(s) in {dname}")

    for trial_dir in tqdm(trials, desc=dname, unit="trial"):
        tqdm.write(f"  {trial_dir.name}")
        outdir = output_root / dname / "test" / trial_dir.name
        process_trial(
            dataset_type   = args.dataset_type,
            trial_dir      = trial_dir,
            outdir         = outdir,
            views          = views,
            mvtracker_model= mvtracker_model,
            device         = device,
            depth_subdir   = depth_subdir,
            depth_type     = depth_type,
            max_queries    = args.max_queries,
            query_seed     = args.query_seed,
            trial_index    = _order.get(trial_dir.name, 0),
            use_cached_tracks = not args.no_cached_tracks,
            overwrite      = args.overwrite,
            posetail_root  = Path(args.posetail_root) if args.posetail_root else None,
        )

    print("Done.")


if __name__ == "__main__":
    main()
