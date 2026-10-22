"""
Convert Point Odyssey GT depth PNGs to the stacked-npz format expected by
mvtracker_inference.py (same as DUSt3R precomputed depths).

Source layout:
    <raw_root>/<session>/view<N>/depths/depth_<frame:05d>.png
    (uint16 PNG, shape (H, W))

Output layout:
    <output_root>/<session>/trial/duster-views-012345678/3d_model__{frame:05d}__scene.npz
    depths key: (C, H, W) float32

Usage:
    python preprocess_gt_depths_point_odyssey.py \\
        --raw-root   /groups/karashchuk/karashchuklab/animal-datasets/point-odyssey/results/test_animal \\
        --output-root /groups/karashchuk/karashchuklab/animal-datasets-processed/po-animal/point-odyssey-animal/test \\
        --depth-scale 1.0
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", required=True,
                   help="Path to raw PO test split, e.g. .../results/test_animal")
    p.add_argument("--output-root", required=True,
                   help="Path to processed dataset test dir, e.g. .../point-odyssey-animal/test")
    p.add_argument("--depth-scale", type=float, default=1.0,
                   help="Multiply uint16 values by this to get float32 depth (default: 1.0)")
    p.add_argument("--n-views", type=int, default=9,
                   help="Number of camera views (default: 9 → duster-views-012345678)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing .npz files")
    return p.parse_args()


def main():
    args = parse_args()

    raw_root    = Path(args.raw_root)
    output_root = Path(args.output_root)
    subdir_name = "gt-depths"
    print(f"Depth subdir: {subdir_name}")

    sessions = sorted(d.name for d in raw_root.iterdir() if d.is_dir())
    print(f"Found {len(sessions)} sessions: {sessions}")

    for session in tqdm(sessions, desc="sessions"):
        # Gather frame list from view0
        view0_depth_dir = raw_root / session / "view0" / "depths"
        depth_files = sorted(view0_depth_dir.glob("depth_*.png"))
        n_frames = len(depth_files)

        out_subdir = output_root / session / "trial" / subdir_name
        out_subdir.mkdir(parents=True, exist_ok=True)

        for frame_idx in tqdm(range(n_frames), desc=session, leave=False):
            out_path = out_subdir / f"3d_model__{frame_idx:05d}__scene.npz"
            if out_path.exists() and not args.overwrite:
                continue

            # Read all camera views for this frame
            depths_per_view = []
            for v in range(args.n_views):
                png_path = raw_root / session / f"view{v}" / "depths" / f"depth_{frame_idx:05d}.png"
                img = Image.open(png_path)
                depth = np.array(img, dtype=np.float32) * args.depth_scale  # (H, W)
                depths_per_view.append(depth)

            depths = np.stack(depths_per_view, axis=0)  # (C, H, W)
            np.savez(str(out_path), depths=depths)

        print(f"  {session}: {n_frames} frames → {out_subdir}")

    print("Done.")


if __name__ == "__main__":
    main()
