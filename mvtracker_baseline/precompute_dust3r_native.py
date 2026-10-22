"""
Precompute DUSt3R depth maps for MVTracker-native datasets.

Supports: kubric-multiview, dex-ycb-multiview, panoptic-multiview

Writes per-frame depth files alongside the scene data:
    <scene_root>/duster-views-{i}-{j}-.../{frame:05d}__scene.npz

Camera subsets use default_rng(view_seed).choice(total, n_views, replace=False)
so that precompute and mvtracker_inference_native.py always agree on the same subset.

Usage:
    cd ~
    export PYTHONPATH="/home/ruppk2@hhmi.org/software/mvtracker:/home/ruppk2@hhmi.org/software/dust3r:$PYTHONPATH"
    /home/ruppk2@hhmi.org/miniforge3/envs/mvtracker/bin/python \\
        /home/ruppk2@hhmi.org/posetail/mvtracker_baseline/precompute_dust3r_native.py \\
        --dataset-type kubric \\
        --data-root /groups/karashchuk/karashchuklab/mvtracker-datasets \\
        --checkpoint /groups/karashchuk/karashchuklab/mvtracker-datasets/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
        --device cuda:0 \\
        --n-views 4 \\
        --view-seed 0
"""

import argparse
import json
import logging
import warnings
from copy import deepcopy
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("dust3r").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

import numpy as np
import torch
import torch.nn.functional as F
import kornia.geometry
from tqdm import tqdm

from dust3r.cloud_opt import global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.utils.image import load_images


TOTAL_VIEWS = {"kubric-multiview": 10, "dex-ycb-multiview": 8, "panoptic-multiview": 31}

# Exact camera subsets MVTracker used in its main evaluation table.
# Each entry is a list of view-index lists (one per run), so panoptic can have
# multiple combos while kubric/dex-ycb have exactly one.
MVTRACKER_CANONICAL_VIEWS = {
    "kubric-multiview":   [[0, 1, 2, 3]],
    "dex-ycb-multiview":  [[0, 1, 2, 3]],
    "panoptic-multiview": [[1, 7, 14, 20], [27, 16, 14, 8], [1, 4, 7, 11]],
}


# ── camera loaders ────────────────────────────────────────────────────────────

def load_cameras_kubric(scene_root: Path, views: list):
    """
    Return pixel-space intrinsics (C,3,3) and w2c extrinsics (C,4,4).
    Follows KubricMultiViewDataset.getitem_raw_datapoint coordinate convention:
        K_px  = diag(W, H, 1) @ K_norm @ diag(1, -1, -1)
        E_3x4 = diag(1, -1, -1) @ c2w.inverse()[:3, :]
    """
    intrs_list, extrs_list = [], []
    for v in views:
        with open(scene_root / f"view_{v}" / "metadata.json") as f:
            meta = json.load(f)
        cam  = meta["camera"]
        W, H = meta["metadata"]["resolution"]
        K_px = np.diag([W, H, 1.0]) @ np.array(cam["K"], dtype=np.float64) @ np.diag([1.0, -1.0, -1.0])

        positions   = torch.tensor(cam["positions"],   dtype=torch.float64)
        quaternions = torch.tensor(cam["quaternions"], dtype=torch.float64)
        R   = kornia.geometry.quaternion_to_rotation_matrix(quaternions)
        c2w = torch.zeros(4, 4, dtype=torch.float64)
        c2w[:3, :3] = R[0];  c2w[:3, 3] = positions[0];  c2w[3, 3] = 1.0
        E_3x4 = np.diag([1.0, -1.0, -1.0]) @ c2w.inverse()[:3, :].numpy()
        E_4x4 = np.eye(4, dtype=np.float64);  E_4x4[:3, :] = E_3x4

        intrs_list.append(K_px.astype(np.float32))
        extrs_list.append(E_4x4.astype(np.float32))
    return np.stack(intrs_list), np.stack(extrs_list)


def load_cameras_dexycb(scene_root: Path, views: list):
    """Return (C,3,3) intrinsics and (C,4,4) w2c extrinsics."""
    intrs_list, extrs_list = [], []
    for v in views:
        d = np.load(scene_root / f"view_{v:02d}" / "intrinsics_extrinsics.npz")
        intrs_list.append(np.array(d["intrinsics"], dtype=np.float32)[:3, :3])
        extrs_list.append(np.array(d["extrinsics"], dtype=np.float32))
    return np.stack(intrs_list), np.stack(extrs_list)


def load_cameras_panoptic(scene_root: Path, views: list):
    """Return (C,3,3) intrinsics and (C,4,4) w2c extrinsics from tapvid3d_annotations.npz."""
    ann      = np.load(scene_root / "tapvid3d_annotations.npz")
    intrs_np = np.stack([ann["intrinsics"][v] for v in views]).astype(np.float32)
    extrs_np = np.stack([ann["extrinsics"][v] for v in views]).astype(np.float32)
    if extrs_np.shape[-2] == 3:   # pad (C,3,4) → (C,4,4)
        pad = np.zeros((len(views), 1, 4), dtype=np.float32);  pad[:, 0, 3] = 1.0
        extrs_np = np.concatenate([extrs_np, pad], axis=1)
    return intrs_np, extrs_np


# ── image path loaders ────────────────────────────────────────────────────────

def get_frame_paths_kubric(scene_root: Path, views: list):
    paths = []
    for v in views:
        vp = sorted((scene_root / f"view_{v}").glob("rgba_*.png"))
        if not vp:
            raise FileNotFoundError(f"No rgba_*.png in {scene_root}/view_{v}")
        paths.append(vp)
    return paths


def get_frame_paths_dexycb(scene_root: Path, views: list):
    paths = []
    for v in views:
        vp = sorted((scene_root / f"view_{v:02d}" / "rgb").glob("*.png"))
        if not vp:
            raise FileNotFoundError(f"No png in {scene_root}/view_{v:02d}/rgb")
        paths.append(vp)
    return paths


def get_frame_paths_panoptic(scene_root: Path, views: list):
    paths = []
    for v in views:
        folder = scene_root / "ims" / str(v)
        vp = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
        if not vp:
            raise FileNotFoundError(f"No images in {folder}")
        paths.append(vp)
    return paths


CAMERA_LOADERS    = {"kubric-multiview":   load_cameras_kubric,
                     "dex-ycb-multiview":  load_cameras_dexycb,
                     "panoptic-multiview": load_cameras_panoptic}
FRAMEPATH_LOADERS = {"kubric-multiview":   get_frame_paths_kubric,
                     "dex-ycb-multiview":  get_frame_paths_dexycb,
                     "panoptic-multiview": get_frame_paths_panoptic}


# ── scene discovery ───────────────────────────────────────────────────────────

def find_scenes(data_root: Path, dataset_type: str):
    subdirs = {
        "kubric-multiview":   data_root / "kubric-multiview"  / "test",
        "dex-ycb-multiview":  data_root / "dex-ycb-multiview",
        "panoptic-multiview": data_root / "panoptic-multiview",
    }
    base = subdirs[dataset_type]
    if not base.exists():
        raise FileNotFoundError(f"{base} not found")
    return sorted([d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")])


# ── shared DUSt3R processing loop ─────────────────────────────────────────────

def process_scene(dataset_type: str, scene_root: Path, views: list, model, device,
                  image_size=512, ga_niter=300, ga_lr=0.01, ga_schedule="linear",
                  min_conf_thr=20, clean_depth=True, skip_existing=True):
    if len(views) < 2:
        print(f"  [skip] DUSt3R requires at least 2 views (got {len(views)})")
        return 0

    views_str   = "-".join(str(v) for v in views)
    output_path = scene_root / f"duster-views-{views_str}"
    n_views     = len(views)

    print(f"  {scene_root.name}  views={views}  → {output_path.name}")

    try:
        intrs_np, extrs_np = CAMERA_LOADERS[dataset_type](scene_root, views)
        frame_paths        = FRAMEPATH_LOADERS[dataset_type](scene_root, views)
    except (FileNotFoundError, KeyError) as e:
        print(f"    [skip] {e}")
        return 0

    n_frames = len(frame_paths[0])
    if not all(len(fp) == n_frames for fp in frame_paths):
        print("    [skip] frame count mismatch across views")
        return 0

    output_path.mkdir(exist_ok=True)

    fx = torch.tensor(intrs_np[:, 0, 0])
    fy = torch.tensor(intrs_np[:, 1, 1])
    cx = torch.tensor(intrs_np[:, 0, 2])
    cy = torch.tensor(intrs_np[:, 1, 2])
    extrinsics   = torch.from_numpy(extrs_np).float()
    known_poses  = extrinsics.inverse()
    known_focals = (fx + fy) / 2
    known_pp     = torch.stack([cx, cy], dim=-1)

    from PIL import Image as PILImage
    _img0 = PILImage.open(str(frame_paths[0][0]))
    original_w, original_h = _img0.size
    _img0.close()

    images_all = []
    target_w = target_h = None
    for frame_idx in range(n_frames):
        paths  = [str(frame_paths[vi][frame_idx]) for vi in range(n_views)]
        images = load_images(paths, image_size, verbose=False)
        if target_w is None:
            target_h = int(images[0]["true_shape"][0][0])
            target_w = int(images[0]["true_shape"][0][1])
        images_all.append(images)

    patch_h, patch_w = model.patch_embed.patch_size
    pad_h = (patch_h - target_h % patch_h) % patch_h
    pad_w = (patch_w - target_w % patch_w) % patch_w
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

    if pad_h or pad_w:
        for frame_images in images_all:
            for im in frame_images:
                im["img"] = F.pad(im["img"],
                                  (pad_left, pad_right, pad_top, pad_bottom),
                                  mode="replicate")
                im["true_shape"] = np.int32([[target_h + pad_h, target_w + pad_w]])
        known_pp = known_pp.clone()
        known_pp[..., 0] += pad_left
        known_pp[..., 1] += pad_top

    rescale = (target_w / original_w) if target_w == 512 else (target_h / original_h)
    t0, l0 = pad_top, pad_left
    b0, r0 = t0 + target_h, l0 + target_w

    n_saved = 0
    for frame_idx in tqdm(range(n_frames), desc=f"    frames", unit="frame", leave=False):
        out_file = output_path / f"3d_model__{frame_idx:05d}__scene.npz"
        if skip_existing and out_file.exists():
            try:
                np.load(out_file);  continue
            except Exception:
                pass

        images = images_all[frame_idx]
        pairs  = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        out    = inference(pairs, model, device, batch_size=1, verbose=False)

        if pad_h or pad_w:
            out["view1"]["img"]   = out["view1"]["img"][:, :, t0:b0, l0:r0].contiguous()
            out["view2"]["img"]   = out["view2"]["img"][:, :, t0:b0, l0:r0].contiguous()
            out["pred1"]["conf"]  = out["pred1"]["conf"][:, t0:b0, l0:r0].contiguous()
            out["pred2"]["conf"]  = out["pred2"]["conf"][:, t0:b0, l0:r0].contiguous()
            out["pred1"]["pts3d"] = out["pred1"]["pts3d"][:, t0:b0, l0:r0, :].contiguous()
            out["pred2"]["pts3d_in_other_view"] = (
                out["pred2"]["pts3d_in_other_view"][:, t0:b0, l0:r0, :].contiguous())
            out["view1"]["true_shape"] = np.int32([[target_h, target_w]])
            out["view2"]["true_shape"] = np.int32([[target_h, target_w]])

        scene = global_aligner(out, device=device, verbose=False)
        scene.preset_focal(known_focals.clone() * rescale)
        scene.im_pp.requires_grad_(True)
        scene.preset_principal_point(known_pp.clone() * rescale)
        scene.preset_pose(known_poses.clone())

        scene.compute_global_alignment(
            init="known_poses", niter=ga_niter, schedule=ga_schedule, lr=ga_lr)

        if clean_depth:
            scene = scene.clean_pointcloud()

        depths = np.stack(to_numpy(scene.get_depthmaps()))
        confs  = np.stack(to_numpy(list(scene.im_conf)))
        scene_cp = deepcopy(scene)
        scene_cp.min_conf_thr = float(scene_cp.conf_trf(torch.tensor(float(min_conf_thr))))
        msk = np.stack(to_numpy(scene_cp.get_masks()))

        feats_list = []
        for vi in range(n_views):
            with torch.no_grad():
                feat, _, _ = model._encode_image(
                    images[vi]["img"].to(device), images[vi]["true_shape"])
            feats_list.append(feat.detach().cpu().numpy())
        feats = np.concatenate(feats_list, axis=0)

        np.savez(str(out_file), depths=depths, confs=confs, cleaned_mask=msk,
                 feats=feats, min_conf_thr=min_conf_thr, clean_depth=clean_depth)
        n_saved += 1

    torch.cuda.empty_cache()
    if n_saved == 0:
        print(f"    → already complete, skipping")
    else:
        print(f"    → {n_saved} new frame(s) saved")
    return n_saved


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-type", required=True,
                   choices=list(TOTAL_VIEWS),
                   help="Match the actual directory name under --data-root")
    p.add_argument("--data-root", required=True,
                   help="Root containing kubric-multiview/, dex-ycb-multiview/, panoptic-multiview/")
    p.add_argument("--checkpoint", required=True,
                   help="Path to DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-views", type=int, default=None,
                   help="Use N randomly-selected views; None = all views")
    p.add_argument("--view-seed", type=int, default=0,
                   help="RNG seed for view subsampling (must match mvtracker_inference_native.py seed)")
    p.add_argument("--views", type=int, nargs="+", default=None,
                   help="Explicit view indices to use; overrides --n-views and --view-seed")
    p.add_argument("--mvtracker-canonical", action="store_true",
                   help="Use MVTRACKER_CANONICAL_VIEWS for this dataset; overrides all other "
                        "view-selection flags")
    p.add_argument("--ga-niter", type=int, default=300)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    total  = TOTAL_VIEWS[args.dataset_type]

    if args.mvtracker_canonical:
        views_list = MVTRACKER_CANONICAL_VIEWS[args.dataset_type]
        print(f"MVTracker canonical views ({len(views_list)} combo(s)): {views_list}")
    elif args.views is not None:
        views_list = [sorted(args.views)]
        print(f"Explicit views: {views_list[0]}")
    elif args.n_views is not None and total > args.n_views:
        rng        = np.random.default_rng(args.view_seed)
        views_list = [sorted(rng.choice(total, args.n_views, replace=False).tolist())]
        print(f"Subsampled {args.n_views}/{total} views: {views_list[0]}")
    else:
        views_list = [list(range(total))]
        print(f"Using all {total} views")

    for views in views_list:
        print(f"  Depth subdir: duster-views-{'-'.join(str(v) for v in views)}")

    print(f"\nLoading DUSt3R from {args.checkpoint} …")
    model = AsymmetricCroCo3DStereo.from_pretrained(args.checkpoint).to(device)
    model.eval()
    print("DUSt3R loaded.\n")

    scenes = find_scenes(Path(args.data_root), args.dataset_type)
    print(f"Found {len(scenes)} scene(s)\n")

    for scene in tqdm(scenes, desc=args.dataset_type, unit="scene"):
        for views in views_list:
            process_scene(
                dataset_type  = args.dataset_type,
                scene_root    = scene,
                views         = views,
                model         = model,
                device        = device,
                ga_niter      = args.ga_niter,
                skip_existing = not args.overwrite,
            )

    print("Done.")


if __name__ == "__main__":
    main()
