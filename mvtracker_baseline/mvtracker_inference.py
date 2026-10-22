"""
Run MVTracker inference on all trials in a dataset split.

MVTracker is a native multi-view 3D tracker: it takes all camera views
simultaneously and outputs 3D tracks directly — no triangulation step needed.

Depth maps are estimated per frame using DUSt3R. The MVTracker repo must be
in PYTHONPATH. Clone it and install its requirements first:

    git clone https://github.com/ethz-vlg/mvtracker.git
    pip install -r mvtracker/requirements.full.txt
    export PYTHONPATH="/path/to/mvtracker:$PYTHONPATH"

Download the checkpoint:
    wget https://huggingface.co/ethz-vlg/mvtracker/resolve/main/mvtracker_200000_june2025.pth \\
         -P checkpoints/

Outputs are saved to:
    <output_root>/dataset/split/session/trial/mvtracker_3d.npz

The output .npz contains the same fields as cotracker_3d.npz so
cotracker_metrics.py can evaluate it with --prefix mvtracker_:
    coords_3d     (T, N, 3)  float32   predicted 3D world coords
    visibility    (T, N)     bool      MVTracker visibility predictions
    frame_numbers (T,)       int64     absolute frame indices
    valid_mask    (S*K,)     bool      which (subject×kpt) entries are included
    cam_names     (C,)       str       camera names used

Example
-------
    python mvtracker_inference.py \\
        --dataset-root /data/posetail-pretraining-v5 \\
        --output-root  /home/user/mvtracker-outputs \\
        --datasets kubric-multiview dex_ycb cmupanoptic_3dgs \\
        --split test \\
        --checkpoint checkpoints/mvtracker_200000_june2025.pth \\
        --device cuda:0
"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

try:
    from tracktail.datasets.utils import get_dirs
    from tracktail.inference.inference_utils import (
        build_video_readers,
        compute_query_first,
        load_multiview_clip,
    )
except ImportError:   # checkout that still uses the old package name (no compute_query_first / --query-first)
    from posetail.datasets.utils import get_dirs
    from posetail.inference.inference_utils import (
        build_video_readers,
        load_multiview_clip,
    )

# MVTracker's own per-dataset evaluation settings (configs/model/mvtracker.yaml, evaluation.predictor_settings),
# used with --official-settings. Keyed by the posetail dataset name.
OFFICIAL_PREDICTOR_SETTINGS = {
    'kubric-multiview': dict(visibility_threshold=0.5,  grid_size=4, local_grid_size=18),
    'dex_ycb':          dict(visibility_threshold=0.01, grid_size=4, local_grid_size=18),
    'cmupanoptic_3dgs': dict(visibility_threshold=0.01, grid_size=6, local_grid_size=18),
    # point odyssey has no settings of its own: MVTracker's generic ones (configs/model/mvtracker.yaml, `generic`)
    'point-odyssey-human':     dict(visibility_threshold=0.01, grid_size=4, local_grid_size=18),
    'po-animal-sub256-motion': dict(visibility_threshold=0.01, grid_size=4, local_grid_size=18),
}
# MVTracker's dataloaders apply a per-dataset similarity transform X' = R (s X) + t to the scene (kubric: none).
OFFICIAL_N_ITERS = 4


def official_scene_norm(dataset_name):
    """(scale, rotation (3,3), translation (3,)) of MVTracker's scene normalization, or None."""
    from scipy.spatial.transform import Rotation as R
    if dataset_name == 'cmupanoptic_3dgs':
        return 2.5, R.from_euler('x', -90, degrees=True).as_matrix(), np.array([0.0, 0.0, 0.0])
    if dataset_name == 'dex_ycb':
        rot = (R.from_euler('z', -30, degrees=True).as_matrix()
               @ R.from_euler('y', 3, degrees=True).as_matrix()
               @ R.from_euler('x', 220, degrees=True).as_matrix())
        return 6.0, rot, np.array([0.0, 0.0, 0.5])
    return None


def build_model(checkpoint, device, dataset_name=None, official_settings=False, vis_threshold=None):
    """MVTracker from torch.hub; with official_settings the predictor uses MVTracker's per-dataset settings.
    vis_threshold replaces the visibility threshold of those settings (a track is visible when its probability of
    being visible is above it)."""
    if official_settings:
        kwargs = dict(OFFICIAL_PREDICTOR_SETTINGS[dataset_name], n_iters=OFFICIAL_N_ITERS)
        if vis_threshold is not None:
            kwargs['visibility_threshold'] = vis_threshold
        print(f"Loading MVTracker with the official settings for {dataset_name}: {kwargs}")
        model = torch.hub.load("ethz-vlg/mvtracker", "mvtracker_predictor", pretrained=False,
                               device=str(device), predictor_kwargs=kwargs)
    else:
        print("Loading MVTracker…")
        model = torch.hub.load("ethz-vlg/mvtracker", "mvtracker", pretrained=False, device=device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    # Checkpoint keys have no "model." prefix but EvaluationPredictor stores
    # the inner MVTracker as self.model — load into the inner model directly.
    model.model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    return model

# ── camera loading ────────────────────────────────────────────────────────────

def load_camera_params(metadata_path: str):
    """
    Load camera intrinsics and extrinsics from metadata.yaml.

    Returns
    -------
    cam_names : list[str]
    intrs_np  : (C, 3, 3) float32  – camera matrices
    extrs_np  : (C, 4, 4) float32  – world-to-camera matrices (static cameras)
                or (T, C, 4, 4)    – per-frame world-to-camera (moving cameras)
    moving    : bool
    """
    with open(metadata_path, "r") as f:
        meta = yaml.safe_load(f)

    intrinsics_dict  = meta["intrinsic_matrices"]
    extrinsics_dict  = meta["extrinsic_matrices"]

    cam_names = list(intrinsics_dict.keys())
    if all(n.isdigit() for n in cam_names):
        cam_names = sorted(cam_names, key=int)
    else:
        cam_names = sorted(cam_names)

    intrs_np = np.stack(
        [np.asarray(intrinsics_dict[n], dtype=np.float32) for n in cam_names],
        axis=0,
    )  # (C, 3, 3)

    moving = bool(meta.get("moving_cams", False))

    if moving:
        # (T, C, 4, 4): per-frame extrinsics stored as (T, 4, 4) per camera
        ext_per_cam = [
            np.asarray(extrinsics_dict[n], dtype=np.float32)  # (T, 4, 4)
            for n in cam_names
        ]
        extrs_np = np.stack(ext_per_cam, axis=1)  # (T, C, 4, 4)
    else:
        extrs_np = np.stack(
            [np.asarray(extrinsics_dict[n], dtype=np.float32) for n in cam_names],
            axis=0,
        )  # (C, 4, 4)

    return cam_names, intrs_np, extrs_np, moving


def build_mvtracker_cameras(intrs_np, extrs_np, T: int, moving: bool):
    """
    Tile static camera params across T frames to produce MVTracker input shapes.

    Returns
    -------
    intrs_t : (T, C, 3, 3) float32
    extrs_t : (T, C, 4, 4) float32
    """
    # intrs are always time-invariant
    intrs_t = np.broadcast_to(intrs_np[np.newaxis], (T,) + intrs_np.shape).copy()

    if moving:
        # extrs_np already (T, C, 4, 4); clip/pad to match actual clip length
        extrs_t = extrs_np[:T]
    else:
        extrs_t = np.broadcast_to(extrs_np[np.newaxis], (T,) + extrs_np.shape).copy()

    return intrs_t.astype(np.float32), extrs_t.astype(np.float32)


# ── DUSt3R depth loading ──────────────────────────────────────────────────────
#
# Depths are precomputed by precompute_dust3r_depths.py and stored as:
#   <depth_root>/<dataset>/<split>/<session>/<trial>/duster-views-{views}/3d_model__{frame:05d}__scene.npz
# Each .npz has a 'depths' key with shape (C, H, W) float32.

def load_precomputed_depths(depth_dir: str, frame_numbers: np.ndarray, H: int, W: int,
                            depth_subdir: str = None) -> np.ndarray:
    """
    Load per-frame precomputed depths from __scene.npz files.

    Parameters
    ----------
    depth_dir     : trial-level directory
    frame_numbers : (T,) absolute frame indices to load
    H, W          : expected spatial resolution for depth (image resolution)
    depth_subdir  : subdirectory name within depth_dir containing the .npz files
                    (e.g. 'duster-views-012345678' or 'depths')

    Returns
    -------
    depths : (T, C, H, W) float32
    """
    depth_dir = Path(depth_dir)
    if depth_subdir:
        depth_dir = depth_dir / depth_subdir

    depths_list = []
    for frame_idx in frame_numbers:
        npz_path = depth_dir / f"3d_model__{int(frame_idx):05d}__scene.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Precomputed depth not found: {npz_path}\n"
                f"Run precompute_dust3r_depths.py first."
            )
        data   = np.load(npz_path)
        depths = data["depths"].astype(np.float32)   # (C, H_d, W_d)
        if depths.shape[-2:] != (H, W):
            import cv2
            C = depths.shape[0]
            resized = np.stack([cv2.resize(depths[c], (W, H)) for c in range(C)])
            depths = resized
        depths_list.append(depths)

    return np.stack(depths_list)   # (T, C, H, W)


def load_native_gt_depths(dataset_name, native_trial_dir, cam_names, frame_numbers, H, W):
    """(T, C, H, W) GT depths of the cameras cam_names, read from the dataset's own files by the loaders of
    mvtracker_inference_native_official.py (kubric: euclidean tiffs -> z-depth; panoptic: dynamic3dgs_depth)."""
    import mvtracker_inference_native_official as native
    views = [int(str(n).replace("view_", "")) for n in cam_names]
    if dataset_name == "kubric-multiview":
        depths = native.load_kubric_gt_depths(Path(native_trial_dir), views, frame_numbers)
    elif dataset_name == "cmupanoptic_3dgs":
        depths = native.load_panoptic_gt_depths(Path(native_trial_dir), views, frame_numbers)
    else:
        raise ValueError(f"no GT depths for {dataset_name!r}")
    if depths.shape[-2:] != (H, W):
        depths = np.stack([[cv2.resize(d, (W, H)) for d in frame] for frame in depths]).astype(np.float32)
    return depths


# ── query loading ─────────────────────────────────────────────────────────────

def load_query_points(pose_path: str, start_frame: int = 0,
                      max_queries: int | None = None, query_seed: int = 0,
                      query_first: bool = False, vis_cams=None):
    """
    Load 3D query points from pose3d.npz.

    With query_first each point is queried at its first valid frame (finite coords and visible in at least one
    camera of the pose file, as in tracktail's query_first) instead of at start_frame, e.g. for pose3d_cached.npz.
    vis_cams restricts that visibility to those columns of the pose file's vis (the cameras that are used).

    Returns
    -------
    query_3d    : (N, 3) float32  – valid 3D coords at the query frame
    valid_mask  : (S*K,) bool     – which (subject × kpt) entries are included
    query_times : (N,) int64      – query frame of each point, relative to start_frame
    """
    data = np.load(pose_path)
    coords = data["pose"]                          # (S, T, K, 3)
    S, T, K, _ = coords.shape
    if query_first:
        vis = data["vis"] if "vis" in data else None
        if vis is not None and vis_cams is not None:
            vis = vis[..., vis_cams]
        queries, times, valids = [], [], []
        for s in range(S):
            qt_s, qc_s, valid_s = compute_query_first(
                coords[s], vis[s] if vis is not None else None, start_frame, None)
            queries.append(qc_s)
            times.append(qt_s)
            valids.append(valid_s)
        coords_flat = np.concatenate(queries)         # (S*K, 3) at each point's query frame
        times_flat = np.concatenate(times).astype(np.int64)
        valid = np.concatenate(valids)
    else:
        coords_at_start = coords[:, start_frame, :, :]   # (S, K, 3)
        coords_flat = coords_at_start.reshape(-1, 3)      # (S*K, 3)
        times_flat = np.zeros(len(coords_flat), dtype=np.int64)
        valid = np.all(np.isfinite(coords_flat), axis=1)

    # Subsample if there are more valid points than max_queries
    # (Point Odyssey has ~45k points; MVTracker handles ~384)
    if max_queries is not None:
        valid_indices = np.where(valid)[0]
        if len(valid_indices) > max_queries:
            rng = np.random.RandomState(query_seed)
            chosen = rng.choice(valid_indices, max_queries, replace=False)
            chosen.sort()
            valid = np.zeros(len(valid), dtype=bool)
            valid[chosen] = True

    query_3d = coords_flat[valid].astype(np.float32)
    return query_3d, valid, times_flat[valid]


# ── per-trial processing ──────────────────────────────────────────────────────

def process_trial(
    trial_path: str,
    out_path: str,
    mvtracker_model,
    device: torch.device,
    depth_dir: str,
    depth_subdir: str = None,
    depth_type: str = "duster",
    start_frame: int = 0,
    n_frames: int | None = None,
    max_res: int | None = None,
    cam_names_filter: list[str] | None = None,
    n_views: int | None = None,
    view_seed: int = 0,
    max_queries: int | None = None,
    query_seed: int = 0,
    query_chunk_size: int | None = None,
    overwrite: bool = False,
    pose_file: str | None = None,
    query_first: bool = False,
    dataset_name: str | None = None,
    official_settings: bool = False,
    native_trial_dir: str | None = None,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    metadata_path = os.path.join(trial_path, "metadata.yaml")
    if pose_file is not None:
        pose_path = pose_file if os.path.isabs(pose_file) else os.path.join(trial_path, pose_file)
    else:
        pose_path = os.path.join(trial_path, "pose3d.npz")

    if not os.path.exists(metadata_path):
        print("    [skip] missing metadata.yaml")
        return
    if not os.path.exists(pose_path):
        print("    [skip] missing pose3d.npz")
        return

    cam_names, intrs_np, extrs_np, moving = load_camera_params(metadata_path)

    cam_names_all    = cam_names[:]
    cam_indices_used = list(range(len(cam_names_all)))

    if cam_names_filter is not None:
        missing = [n for n in cam_names_filter if n not in cam_names_all]
        if missing:
            print(f"    [skip] requested cameras not in metadata: {missing}")
            return
        cam_indices_used = [cam_names_all.index(n) for n in cam_names_filter]
        cam_names        = cam_names_filter
        intrs_np         = intrs_np[cam_indices_used]
        extrs_np         = (extrs_np[:, cam_indices_used] if moving else extrs_np[cam_indices_used])
        print(f"    using explicit cameras: {cam_names}")
    elif n_views is not None and len(cam_names_all) > n_views:
        rng              = np.random.default_rng(view_seed)
        cam_indices_used = sorted(rng.choice(len(cam_names_all), n_views, replace=False).tolist())
        cam_names        = [cam_names_all[i] for i in cam_indices_used]
        intrs_np         = intrs_np[cam_indices_used]
        extrs_np         = (extrs_np[:, cam_indices_used] if moving else extrs_np[cam_indices_used])
        print(f"    subsampled {n_views}/{len(cam_names_all)} cams: "
              f"indices={cam_indices_used} names={cam_names}")
    elif n_views is not None:
        print(f"    --n-views={n_views} >= available cameras ({len(cam_names_all)}); using all")

    # Auto-derive depth_subdir from the actual camera selection if not explicitly set.
    # For numeric camera names (e.g. panoptic cam IDs like '1','7','14','20') use the
    # name values so the subdir matches precompute_dust3r_native.py output (duster-views-1-7-14-20).
    # For non-numeric names (view_00, view_0, …) use list positions as before.
    if depth_type == "gt-native":
        depth_subdir = None   # GT depths are read from the dataset's own files, no depth subdir
    elif depth_subdir is None:
        if depth_type == "gt":
            depth_subdir = "gt-depths"
        else:
            if all(n.isdigit() for n in cam_names):
                views_str = "-".join(str(n) for n in cam_names)
            else:
                views_str = "-".join(str(i) for i in cam_indices_used)
            depth_subdir = f"duster-views-{views_str}"

    depth_subdir_path = os.path.join(depth_dir, depth_subdir) if depth_subdir else None
    print(f"    depth_subdir: {depth_subdir}")
    if depth_subdir_path is not None and not os.path.isdir(depth_subdir_path):
        print(f"    [skip] depth subdir not found: {depth_subdir_path}")
        if depth_type == "gt":
            print(f"           Run preprocess_gt_depths_point_odyssey.py --n-views {len(cam_indices_used)} first.")
        else:
            print(f"           Run precompute_dust3r_depths.py --n-views {len(cam_indices_used)} --view-seed {view_seed} first.")
        return

    if os.path.exists(out_path) and not overwrite:
        print("    [skip] already done (--overwrite to redo)")
        return

    # Locate video data
    img_path = os.path.join(trial_path, "img")
    vid_path = os.path.join(trial_path, "vid")
    if os.path.exists(img_path) and os.listdir(img_path):
        video_paths = [os.path.join(img_path, n) for n in cam_names]
    elif os.path.exists(vid_path):
        video_paths = [os.path.join(vid_path, f"{n}.mp4") for n in cam_names]
    else:
        print("    [skip] no img/ or vid/ folder")
        return

    readers, lengths = build_video_readers(video_paths)
    max_available = min(lengths)
    if query_first:
        # some image folders have more frames than the ground truth: stop where the pose file stops
        max_available = min(max_available, np.load(pose_path)["pose"].shape[1])
    end_frame = min(
        start_frame + n_frames if n_frames is not None else max_available,
        max_available,
    )
    T = end_frame - start_frame
    frame_numbers = np.arange(start_frame, end_frame, dtype=np.int64)

    # Load all frames: list of (T, H, W, 3) tensors per camera → stack to (T, C, H, W, 3)
    views, _ = load_multiview_clip(readers, start_frame, T)
    del readers

    C = len(cam_names)
    H_orig, W_orig = views[0].shape[1], views[0].shape[2]

    # Stack cameras: (T, C, H, W, 3) uint8
    frames_all = np.stack(
        [v.numpy() if isinstance(v, torch.Tensor) else v for v in views],
        axis=1,
    ).astype(np.uint8)

    # Load query points
    if query_first and len(cam_indices_used) < len(cam_names_all):
        with np.load(pose_path) as pose_data:
            n_vis = pose_data["vis"].shape[-1] if "vis" in pose_data.files else None
        if n_vis is not None and n_vis != len(cam_names_all):
            raise ValueError(f"{pose_path}: vis has {n_vis} columns but metadata.yaml has {len(cam_names_all)} "
                             f"cameras, so the vis columns cannot be matched to the cameras in use")
    query_3d, valid_mask, query_times = load_query_points(
        pose_path, start_frame=start_frame, max_queries=max_queries, query_seed=query_seed,
        query_first=query_first,
        vis_cams=cam_indices_used if (query_first and len(cam_indices_used) < len(cam_names_all)) else None)
    if len(query_3d) == 0:
        print(f"    [skip] no finite 3D coords at frame {start_frame}")
        return

    # Load precomputed depth maps at native resolution
    print(f"    loading precomputed depths ({T} frames × {C} cams)…")
    if depth_type == "gt-native":
        depths_np = load_native_gt_depths(dataset_name, native_trial_dir, cam_names, frame_numbers, H_orig, W_orig)
    else:
        depths_np = load_precomputed_depths(depth_dir, frame_numbers, H_orig, W_orig, depth_subdir)  # (T, C_file, H_orig, W_orig)

    # Resize and pad frames + depths to TARGET_RES × TARGET_RES (matching posetail's inference resolution)
    TARGET_RES = 256
    rp_scale    = TARGET_RES / max(H_orig, W_orig)
    new_H_rp    = int(round(H_orig * rp_scale))
    new_W_rp    = int(round(W_orig * rp_scale))
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
    # Adjust intrinsics for resize+pad so the model's depth unprojection is correct
    intrs_np[:, 0, :] *= rp_scale    # scale fx and cx
    intrs_np[:, 1, :] *= rp_scale    # scale fy and cy
    intrs_np[:, 0, 2] += pad_left_rp  # shift cx for left pad
    intrs_np[:, 1, 2] += pad_top_rp   # shift cy for top pad
    H, W = TARGET_RES, TARGET_RES
    # GT depths contain all cameras; select only the ones being used
    if depths_np.shape[1] != C:
        depths_np = depths_np[:, cam_indices_used, :, :]   # (T, C, H, W)

    # Scene normalization (--official-settings): MVTracker's dataloaders feed the model a per-dataset
    # similarity transform of the scene; predictions are mapped back to the raw world frame before saving.
    norm = official_scene_norm(dataset_name) if official_settings else None
    if norm is not None:
        from mvtracker.datasets.utils import transform_scene
        scale, rot, trans = norm
        d = torch.from_numpy(depths_np.transpose(1, 0, 2, 3))[:, :, None].float()                  # (C, T, 1, H, W)
        e = torch.from_numpy(np.broadcast_to(extrs_np[:, None, :3, :], (C, T, 3, 4)).copy()).float()
        q = torch.from_numpy(np.concatenate([query_times[:, None].astype(np.float64), query_3d], 1)).float()
        d2, e2, q2, _, _ = transform_scene(scale, torch.from_numpy(rot).float(), torch.from_numpy(trans).float(),
                                           d, e, q, None, None)
        depths_np = d2[:, :, 0].numpy().transpose(1, 0, 2, 3).copy()
        extrs_np = np.concatenate([e2[:, 0].numpy(), np.broadcast_to(np.array([[[0, 0, 0, 1.0]]]), (C, 1, 4))],
                                  axis=1).astype(np.float32)
        query_3d = q2[:, 1:].numpy().astype(np.float32)
        print(f"    scene normalization applied: scale={scale}")

    # Build MVTracker tensors
    intrs_t, extrs_t = build_mvtracker_cameras(intrs_np, extrs_np, T, moving)

    # model convention: (batch, views=C, frames=T, ...)
    # our arrays are (T, C, ...) so we transpose T↔C everywhere

    # frames_all (T,C,H,W,3) → (C,T,3,H,W) → (1,C,T,3,H,W)
    # Model normalizes internally with /255.0; pass float values in [0, 255]
    rgbs_t = torch.from_numpy(
        frames_all.transpose(1, 0, 4, 2, 3).astype(np.float32)
    ).unsqueeze(0).to(device)

    # depths_np (T,C,H,W) → (C,T,H,W) → (1,C,T,1,H,W)
    depths_t = (torch.from_numpy(depths_np.transpose(1, 0, 2, 3))
                .unsqueeze(0)   # (1,C,T,H,W)
                .unsqueeze(3)   # (1,C,T,1,H,W)
                .to(device))

    # intrs_t (T,C,3,3) → (C,T,3,3) → (1,C,T,3,3)
    intrs_t = torch.from_numpy(intrs_t.transpose(1, 0, 2, 3)).unsqueeze(0).to(device)

    # extrs_t (T,C,4,4) → (C,T,4,4) → drop last row → (1,C,T,3,4)
    extrs_t = (torch.from_numpy(extrs_t.transpose(1, 0, 2, 3)[:, :, :3, :])
               .unsqueeze(0).to(device))

    # queries: model expects (1,N,4) = [t_idx, x, y, z]; queries defined at start_frame
    query_t_idx = (query_times[:, None] + start_frame).astype(np.float32)
    query_4d    = np.concatenate([query_t_idx, query_3d], axis=1)  # (N,4)
    queries_t   = torch.from_numpy(query_4d).unsqueeze(0).to(device)  # (1,N,4)

    # Run MVTracker (optionally in query chunks to reduce peak memory)
    N = queries_t.shape[1]
    print(f"    running MVTracker (N={N}, chunk={query_chunk_size})…")
    if query_chunk_size is None or query_chunk_size >= N:
        with torch.no_grad():
            results = mvtracker_model(
                rgbs=rgbs_t,
                depths=depths_t,
                intrs=intrs_t,
                extrs=extrs_t,
                query_points_3d=queries_t,
            )
        traj_e = results["traj_e"].squeeze(0).cpu().numpy()  # (T, N, 3)
        vis_e  = results["vis_e"].squeeze(0).cpu().numpy()   # (T, N)
        vis_prob = results["vis_e_as_prob"].squeeze(0).cpu().numpy() if "vis_e_as_prob" in results else None   # (T, N)
    else:
        traj_chunks, vis_chunks, prob_chunks = [], [], []
        for start in range(0, N, query_chunk_size):
            chunk = queries_t[:, start:start + query_chunk_size]
            with torch.no_grad():
                res = mvtracker_model(
                    rgbs=rgbs_t,
                    depths=depths_t,
                    intrs=intrs_t,
                    extrs=extrs_t,
                    query_points_3d=chunk,
                )
            traj_chunks.append(res["traj_e"].squeeze(0).cpu().numpy())
            vis_chunks.append(res["vis_e"].squeeze(0).cpu().numpy())
            if "vis_e_as_prob" in res:
                prob_chunks.append(res["vis_e_as_prob"].squeeze(0).cpu().numpy())
            torch.cuda.empty_cache()
        traj_e = np.concatenate(traj_chunks, axis=1)  # (T, N, 3)
        vis_e  = np.concatenate(vis_chunks,  axis=1)  # (T, N)
        vis_prob = np.concatenate(prob_chunks, axis=1) if prob_chunks else None

    if norm is not None:
        # X' = R (s X) + t  =>  X = R^T (X' - t) / s
        traj_e = ((traj_e - trans) @ rot) / scale

    # Convert vis_e to float logits for posetail-compatible vis_pred: (1, T, N, 1)
    if vis_e.dtype == bool or vis_e.dtype == np.bool_:
        vis_pred = np.where(vis_e, 10.0, -10.0).astype(np.float32)
    else:
        vis_pred = vis_e.astype(np.float32)
    vis_pred = vis_pred[np.newaxis, :, :, np.newaxis]   # (1, T, N, 1)

    # Load GT for posetail-compatible output (same logic as load_ground_truth in mvtracker_metrics.py)
    pose_data  = np.load(pose_path)
    coords_gt  = pose_data['pose']                                          # (S, T_full, K, 3)
    vis_gt_raw = pose_data['vis'] if 'vis' in pose_data else None
    S_gt, T_full, K_gt, _ = coords_gt.shape
    frame_nums_clipped = np.clip(frame_numbers, 0, T_full - 1)

    coords_flat = coords_gt.transpose(0, 2, 1, 3).reshape(S_gt * K_gt, T_full, 3)
    coords_flat = coords_flat[valid_mask]
    coords_true = coords_flat[:, frame_nums_clipped, :].transpose(1, 0, 2)[np.newaxis]  # (1, T, N, 3)

    vis_true_cams = None
    if vis_gt_raw is not None:
        vis_sub  = vis_gt_raw[..., cam_indices_used]                           # (S, T_full, K, C)
        vis_agg  = vis_sub.any(axis=-1)                                        # (S, T_full, K)
        vis_flat = vis_agg.transpose(0, 2, 1).reshape(S_gt * K_gt, T_full)
        vis_flat = vis_flat[valid_mask]
        vis_true = vis_flat[:, frame_nums_clipped].T[np.newaxis, :, :, np.newaxis]  # (1, T, N, 1)

        vis_cams_flat = vis_sub.transpose(0, 2, 1, 3).reshape(S_gt * K_gt, T_full, -1)
        vis_cams_flat = vis_cams_flat[valid_mask]                              # (N, T_full, C)
        vis_true_cams = vis_cams_flat[:, frame_nums_clipped, :].transpose(1, 0, 2)[np.newaxis]  # (1, T, N, C)
    else:
        vis_true = np.all(np.isfinite(coords_true), axis=-1, keepdims=True)   # (1, T, N, 1)

    N = traj_e.shape[1]
    save_dict = dict(
        coords_pred      = traj_e[np.newaxis].astype(np.float32),   # (1, T, N, 3)
        vis_pred         = vis_pred,                                 # (1, T, N, 1)
        conf_pred        = vis_pred[..., 0],                        # (1, T, N)   logits (same as vis_pred)
        **({} if vis_prob is None else {"vis_prob": vis_prob.astype(np.float32)}),   # (T, N) probability of being visible, before the threshold
        coords_true      = coords_true.astype(np.float32),          # (1, T, N, 3)
        vis_true         = vis_true.astype(bool),                   # (1, T, N, 1)
        query_times      = query_times.astype(np.int32),            # (N,) query frame (0 unless --query-first)
        frame_numbers    = frame_numbers,
        valid_mask       = valid_mask,
        cam_names_used   = np.array(cam_names, dtype=str),
        cam_indices_used = np.array(cam_indices_used, dtype=np.int32),
        video_paths      = np.array(video_paths, dtype=str),
        metadata_path    = np.array(metadata_path, dtype=str),
        start_frame      = np.array(start_frame, dtype=np.int64),
    )
    if vis_true_cams is not None:
        save_dict['vis_true_cams'] = vis_true_cams.astype(bool)     # (1, T, N, C)
    np.savez(out_path, **save_dict)
    print(f"    saved: {tuple(traj_e.shape)}  "
          f"cam_indices={cam_indices_used}  cam_names={cam_names}")
    torch.cuda.empty_cache()


# ── main loop ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--depth-root", required=True,
                   help="Root of precomputed depths. "
                        "Mirrors dataset structure: <depth_root>/<dataset>/<split>/<session>/<trial>/")
    p.add_argument("--depth-subdir", default=None,
                   help="Explicit subdir name within each trial's depth dir (overrides --depth-type).")
    p.add_argument("--depth-type", default="duster", choices=["duster", "gt", "gt-native"],
                   help="'gt-native' = GT depths of kubric / panoptic read from the dataset's own files "
                        "(--native-root); "
                        "'duster' = use duster-views-* subdirs (DUSt3R precomputed); "
                        "'gt' = use gt-views-* subdirs (preprocess_gt_depths_point_odyssey.py output). "
                        "Camera sampling logic is identical in both cases.")
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", required=True,
                   help="Path to mvtracker_200000_june2025.pth")
    p.add_argument("--device", default="cuda")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--n-frames", type=int, default=None,
                   help="Max frames per trial; omit for full trial")
    p.add_argument("--max-res", type=int, default=None,
                   help="Downsample longer edge to this many pixels")
    p.add_argument("--cam-names", nargs="+", default=None,
                   help="Explicit camera names to use; overrides --n-views and --view-seed")
    p.add_argument("--n-views", type=int, default=None,
                   help="Use only this many cameras (random subset; for ablations)")
    p.add_argument("--view-seed", type=int, default=0)
    p.add_argument("--max-queries", type=int, default=None,
                   help="Subsample to this many query points (e.g. 512 for Point Odyssey's ~45k)")
    p.add_argument("--query-seed", type=int, default=0)
    p.add_argument("--query-chunk-size", type=int, default=None,
                   help="Process queries in chunks of this size to reduce peak VRAM")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--pose-file", type=str, default=None,
                   help="Override pose3d.npz with this file for query points (e.g. pose3d_100.npz)")
    p.add_argument("--native-root", default=None,
                   help="With --depth-type gt-native: folder of the dataset's own trials, <native_root>/<session>/ "
                        "(kubric: .../kubric-multiview/test, panoptic: .../panoptic-multiview)")
    p.add_argument("--vis-threshold", type=float, default=None,
                   help="With --official-settings: replace the visibility threshold of the official settings of each "
                        "dataset (e.g. 0.5); the probability of being visible is saved as vis_prob in every output")
    p.add_argument("--query-first", action="store_true",
                   help="Query each point at its first valid frame (finite and visible in a camera of the pose file) "
                        "instead of at the first frame; for pose3d_cached.npz")
    p.add_argument("--official-settings", action="store_true",
                   help="Use MVTracker's own per-dataset evaluation settings: visibility threshold, support grid "
                        "size, 4 iterations and scene normalization (needs the official mvtracker on PYTHONPATH)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    mvtracker_model = None if args.official_settings else build_model(args.checkpoint, device)

    for dataset_name in args.datasets:
        if args.official_settings:   # the predictor settings are per dataset
            mvtracker_model = build_model(args.checkpoint, device, dataset_name, official_settings=True,
                                          vis_threshold=args.vis_threshold)
        split_path = os.path.join(args.dataset_root, dataset_name, args.split)
        sessions   = sorted(get_dirs(split_path))

        all_trials = [
            (session, trial, os.path.join(split_path, session, trial))
            for session in sessions
            for trial in sorted(get_dirs(os.path.join(split_path, session)))
        ]

        ds_out = os.path.join(args.output_root, dataset_name)
        os.makedirs(ds_out, exist_ok=True)

        for session, trial, trial_path in tqdm(all_trials, desc=dataset_name, unit="trial"):
            tqdm.write(f"  {session}/{trial}")
            tid      = f"{session}_{trial}"
            out_path = os.path.join(ds_out, f"{tid}.npz")
            depth_dir = os.path.join(
                args.depth_root, dataset_name, args.split, session, trial
            )
            process_trial(
                trial_path=trial_path,
                out_path=out_path,
                mvtracker_model=mvtracker_model,
                device=device,
                depth_dir=depth_dir,
                depth_subdir=args.depth_subdir,
                depth_type=args.depth_type,
                start_frame=args.start_frame,
                n_frames=args.n_frames,
                max_res=args.max_res,
                cam_names_filter=args.cam_names,
                n_views=args.n_views,
                view_seed=args.view_seed,
                max_queries=args.max_queries,
                query_seed=args.query_seed,
                query_chunk_size=args.query_chunk_size,
                overwrite=args.overwrite,
                pose_file=args.pose_file,
                query_first=args.query_first,
                dataset_name=dataset_name,
                official_settings=args.official_settings,
                native_trial_dir=(os.path.join(args.native_root, session) if args.native_root else None),
            )

    print("Done.")


if __name__ == "__main__":
    main()
