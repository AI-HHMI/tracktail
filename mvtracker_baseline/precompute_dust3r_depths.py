"""
Precompute DUSt3R depth maps for all trials in a posetail dataset split.

Reads:
    <dataset_root>/<dataset>/<split>/<session>/<trial>/metadata.yaml
    <dataset_root>/<dataset>/<split>/<session>/<trial>/img/<cam_name>/

Writes per-frame depth files:
    <depth_root>/<dataset>/<split>/<session>/<trial>/3d_model__{frame:05d}__scene.npz
    Each .npz has a 'depths' field of shape (C, H, W) float32.

This is the preprocessing step that mvtracker_inference.py --depth-source precomputed
expects.  Run it once before inference; it skips frames whose .npz already exists.

Requirements:
    export PYTHONPATH="/home/ruppk2@hhmi.org/software/dust3r:$PYTHONPATH"
    export PYTHONPATH="/home/ruppk2@hhmi.org/software/mvtracker:$PYTHONPATH"

Example:
    cd ~
    /home/ruppk2@hhmi.org/miniforge3/envs/mvtracker/bin/python \\
        /home/ruppk2@hhmi.org/posetail/cotracker_baseline/precompute_dust3r_depths.py \\
        --dataset-root /groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v5 \\
        --depth-root   /groups/karashchuk/karashchuklab/mvtracker-datasets/dust3r-depths-posetail \\
        --datasets cmupanoptic_3dgs \\
        --split test \\
        --checkpoint /groups/karashchuk/karashchuklab/mvtracker-datasets/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \\
        --device cuda:0
"""

import argparse
import os
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from dust3r.cloud_opt import global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.utils.image import load_images

from posetail.datasets.utils import get_dirs


def load_camera_params(metadata_path):
    with open(metadata_path, "r") as f:
        meta = yaml.safe_load(f)
    intrinsics_dict = meta["intrinsic_matrices"]
    extrinsics_dict = meta["extrinsic_matrices"]
    cam_names = list(intrinsics_dict.keys())
    if all(str(n).isdigit() for n in cam_names):
        cam_names = sorted(cam_names, key=int)
    else:
        cam_names = sorted(cam_names)
    intrs_np = np.stack(
        [np.asarray(intrinsics_dict[n], dtype=np.float32) for n in cam_names], axis=0
    )  # (C, 3, 3)
    extrs_np = np.stack(
        [np.asarray(extrinsics_dict[n], dtype=np.float32) for n in cam_names], axis=0
    )  # (C, 4, 4) w2c
    return cam_names, intrs_np, extrs_np


def get_frame_paths(trial_path, cam_names):
    """Return frame_paths[view_idx][frame_idx] from img/<cam>/ directories."""
    img_root = Path(trial_path) / "img"
    frame_paths = []
    for cam in cam_names:
        cam_dir = img_root / str(cam)
        if not cam_dir.exists():
            raise FileNotFoundError(f"Missing camera dir: {cam_dir}")
        paths = sorted(cam_dir.glob("*.png")) + sorted(cam_dir.glob("*.jpg"))
        if not paths:
            raise FileNotFoundError(f"No images found in {cam_dir}")
        frame_paths.append(paths)
    if len(set(len(p) for p in frame_paths)) != 1:
        raise ValueError(f"Camera frame counts differ: {[len(p) for p in frame_paths]}")
    return frame_paths


def process_trial(
    trial_path,
    output_path,
    model,
    device,
    image_size=512,
    ga_niter=300,
    ga_lr=0.01,
    ga_schedule="linear",
    min_conf_thr=20,
    clean_depth=True,
    skip_existing=True,
    silent=False,
    n_views=None,
    view_seed=0,
    cam_names_filter=None,
):
    metadata_path = Path(trial_path) / "metadata.yaml"
    if not metadata_path.exists():
        print("    [skip] missing metadata.yaml")
        return 0

    cam_names, intrs_np, extrs_np = load_camera_params(metadata_path)

    cam_names_all    = cam_names[:]
    cam_indices_used = list(range(len(cam_names_all)))

    if cam_names_filter is not None:
        missing = [n for n in cam_names_filter if n not in cam_names_all]
        if missing:
            print(f"    [skip] requested cameras not in metadata: {missing}")
            return 0
        cam_indices_used = [cam_names_all.index(n) for n in cam_names_filter]
        cam_names        = list(cam_names_filter)
        intrs_np         = intrs_np[cam_indices_used]
        extrs_np         = extrs_np[cam_indices_used]
        print(f"    Using explicit cameras: indices={cam_indices_used} names={cam_names}")
    elif n_views is not None and len(cam_names_all) > n_views:
        rng              = np.random.default_rng(view_seed)
        cam_indices_used = sorted(rng.choice(len(cam_names_all), n_views, replace=False).tolist())
        cam_names        = [cam_names_all[i] for i in cam_indices_used]
        intrs_np         = intrs_np[cam_indices_used]
        extrs_np         = extrs_np[cam_indices_used]
        print(f"    Subsampled {n_views}/{len(cam_names_all)} cameras: "
              f"indices={cam_indices_used} names={cam_names}")
    elif n_views is not None:
        print(f"    --n-views={n_views} >= available cameras ({len(cam_names_all)}); using all")

    try:
        frame_paths = get_frame_paths(trial_path, cam_names)
    except (FileNotFoundError, ValueError) as e:
        print(f"    [skip] {e}")
        return 0

    n_frames  = len(frame_paths[0])
    n_cams    = len(cam_names)
    if all(n.isdigit() for n in cam_names):
        views_str = "-".join(str(n) for n in cam_names)
    else:
        views_str = "-".join(str(i) for i in cam_indices_used)
    output_path = Path(output_path) / f"duster-views-{views_str}"
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"    Depth subdir: duster-views-{views_str}")

    # Camera tensors (all views, static)
    fx = torch.tensor([intrs_np[c, 0, 0] for c in range(n_cams)], dtype=torch.float32)
    fy = torch.tensor([intrs_np[c, 1, 1] for c in range(n_cams)], dtype=torch.float32)
    cx = torch.tensor([intrs_np[c, 0, 2] for c in range(n_cams)], dtype=torch.float32)
    cy = torch.tensor([intrs_np[c, 1, 2] for c in range(n_cams)], dtype=torch.float32)
    extrinsics = torch.from_numpy(extrs_np).float()

    known_poses  = extrinsics.inverse()
    known_focals = (fx + fy) / 2          # scalar per view; DUSt3R uses isotropic focal
    known_pp     = torch.stack([cx, cy], dim=-1)

    # Get original image size from first frame file (needed for intrinsics rescaling)
    from PIL import Image as PILImage
    _img0 = PILImage.open(str(frame_paths[0][0]))
    original_w, original_h = _img0.size
    _img0.close()

    # Pre-load and resize all images, determine padding once
    images_all = []
    target_w = target_h = None
    for frame_idx in range(n_frames):
        paths = [str(frame_paths[v][frame_idx]) for v in range(n_cams)]
        images = load_images(paths, image_size, verbose=False)
        if target_w is None:
            target_h, target_w = int(images[0]['true_shape'][0][0]), int(images[0]['true_shape'][0][1])
        images_all.append(images)

    patch_h, patch_w = model.patch_embed.patch_size
    pad_h = (patch_h - (target_h % patch_h)) % patch_h
    pad_w = (patch_w - (target_w % patch_w)) % patch_w
    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left

    if pad_h or pad_w:
        for frame_images in images_all:
            for im_dict in frame_images:
                im_dict["img"] = F.pad(
                    im_dict["img"],
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="replicate",
                )
                im_dict["true_shape"] = np.int32([[target_h + pad_h, target_w + pad_w]])
        known_pp = known_pp.clone()
        known_pp[..., 0] += pad_left
        known_pp[..., 1] += pad_top

    if target_w == 512:
        rescale = target_w / original_w
    else:
        rescale = target_h / original_h

    t0, l0 = pad_top, pad_left
    b0, r0 = t0 + target_h, l0 + target_w

    n_saved = 0
    for frame_idx in range(n_frames):
        out_file = output_path / f"3d_model__{frame_idx:05d}__scene.npz"
        if skip_existing and out_file.exists():
            try:
                np.load(out_file)
                continue
            except Exception:
                pass

        if not silent:
            print(f"    frame {frame_idx:05d}/{n_frames}")

        images = images_all[frame_idx]
        pairs  = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        out    = inference(pairs, model, device, batch_size=1, verbose=not silent)

        if pad_h or pad_w:
            out["view1"]["img"]   = out["view1"]["img"][:, :, t0:b0, l0:r0].contiguous()
            out["view2"]["img"]   = out["view2"]["img"][:, :, t0:b0, l0:r0].contiguous()
            out["pred1"]["conf"]  = out["pred1"]["conf"][:, t0:b0, l0:r0].contiguous()
            out["pred2"]["conf"]  = out["pred2"]["conf"][:, t0:b0, l0:r0].contiguous()
            out["pred1"]["pts3d"] = out["pred1"]["pts3d"][:, t0:b0, l0:r0, :].contiguous()
            out["pred2"]["pts3d_in_other_view"] = (
                out["pred2"]["pts3d_in_other_view"][:, t0:b0, l0:r0, :].contiguous()
            )
            out["view1"]["true_shape"] = np.int32([[target_h, target_w]])
            out["view2"]["true_shape"] = np.int32([[target_h, target_w]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scene = global_aligner(out, device=device, verbose=not silent)

        scene.preset_focal(known_focals.clone() * rescale)
        scene.im_pp.requires_grad_(True)
        scene.preset_principal_point(known_pp.clone() * rescale)
        scene.preset_pose(known_poses.clone())

        try:
            scene.compute_global_alignment(
                init="mst", niter=ga_niter, schedule=ga_schedule, lr=ga_lr
            )
        except Exception as e:
            print(f"    mst init failed ({e}), retrying with known_poses")
            scene.compute_global_alignment(
                init="known_poses", niter=ga_niter, schedule=ga_schedule, lr=ga_lr
            )

        if clean_depth:
            scene = scene.clean_pointcloud()

        depths = np.stack(to_numpy(scene.get_depthmaps()))   # (C, H, W)
        confs  = np.stack(to_numpy(list(scene.im_conf)))     # (C, H, W)

        scene_cp = deepcopy(scene)
        scene_cp.min_conf_thr = float(scene_cp.conf_trf(torch.tensor(float(min_conf_thr))))
        msk = np.stack(to_numpy(scene_cp.get_masks()))       # (C, H, W) bool

        np.savez(
            str(out_file),
            depths=depths,
            confs=confs,
            cleaned_mask=msk,
            min_conf_thr=min_conf_thr,
            clean_depth=clean_depth,
        )
        n_saved += 1

    torch.cuda.empty_cache()
    return n_saved


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--depth-root",   required=True,
                   help="Where to write precomputed depth files (mirrors dataset structure)")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", required=True,
                   help="Path to DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    p.add_argument("--device", default="cuda")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--ga-niter", type=int, default=300)
    p.add_argument("--cam-names", nargs="+", default=None,
                   help="Explicit camera names to use; overrides --n-views (must match inference --cam-names)")
    p.add_argument("--n-views", type=int, default=None,
                   help="Use only N randomly-selected cameras (for ablations); None = all")
    p.add_argument("--view-seed", type=int, default=0,
                   help="RNG seed for camera subsampling (must match inference seed)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading DUSt3R from {args.checkpoint} …")
    model = AsymmetricCroCo3DStereo.from_pretrained(args.checkpoint).to(device)
    model.eval()
    print("DUSt3R loaded.")

    for dataset_name in args.datasets:
        split_path = os.path.join(args.dataset_root, dataset_name, args.split)
        sessions   = sorted(get_dirs(split_path))

        all_trials = [
            (session, trial,
             os.path.join(split_path, session, trial),
             os.path.join(args.depth_root, dataset_name, args.split, session, trial))
            for session in sessions
            for trial in sorted(get_dirs(os.path.join(split_path, session)))
        ]

        total_saved = 0
        for session, trial, trial_path, depth_out in tqdm(
            all_trials, desc=dataset_name, unit="trial"
        ):
            tqdm.write(f"  {session}/{trial}")
            n = process_trial(
                trial_path=trial_path,
                output_path=depth_out,
                model=model,
                device=device,
                image_size=args.image_size,
                ga_niter=args.ga_niter,
                skip_existing=not args.overwrite,
                n_views=args.n_views,
                view_seed=args.view_seed,
                cam_names_filter=args.cam_names,
            )
            total_saved += n

        print(f"[{dataset_name}] {total_saved} new depth frames saved.")

    print("Done.")


if __name__ == "__main__":
    main()
