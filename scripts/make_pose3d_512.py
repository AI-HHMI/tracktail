#!/usr/bin/env python
"""Curates 512 tracks per trial of the Point Odyssey test sets, stratified by how fast the tracks move.

pose3d_full.npz of a trial has every track (tens of thousands). This samples 512 of them into
<trial>/pose3d_512.npz next to it, in the format of pose3d.npz (which is never touched).

Tracks. Only tracks seen by the 4 cameras (VIEWS, positions in the camera list of metadata.yaml) are eligible: a
track is visible in a frame when its coordinates are finite and it is visible in at least one of the 4 cameras,
and it needs at least MIN_STEPS consecutive visible frame pairs.

Speed. The mean distance a track moves per visible frame pair (path length over the pairs where it is visible at
both frames, divided by their number), so a track is not ranked slow just because it is occluded often.

Bands. Ranked by speed within each trial, so every trial gets the same mix:
    slow      speed below the 25th percentile             128 tracks
    dynamic   25th to 75th percentile                     256 tracks
    fast      75th to 95th percentile                     128 tracks
The top 5% is left out because it is dominated by tracking artefacts. The tracks of a band are drawn at random
without replacement with a generator seeded by SEED and the trial name, so a rerun gives the same tracks.

pose3d_512.npz holds
    pose (1, T, 512, 3), vis (1, T, 512, C)   the sampled tracks of pose3d_full.npz, sorted by source index
    keypoints (512,)                          the names of the sampled tracks in pose3d_full.npz
    band (512,)                               0 slow, 1 dynamic, 2 fast
    speed (512,)                              mean distance per visible frame pair
    source_index (512,)                       index of the track in pose3d_full.npz

Usage:
    python scripts/make_pose3d_512.py                  # both datasets, all trials
    python scripts/make_pose3d_512.py --dry-run
    python scripts/make_pose3d_512.py --datasets point-odyssey-human --overwrite
"""
import argparse
import os
import zlib

import numpy as np

DATASET_ROOT = '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6'
DATASETS = ['point-odyssey-human', 'po-animal-sub256-motion']

SEED = 0
VIEWS = [2, 4, 5, 7]                     # positions in the camera list of metadata.yaml (view2, view4, view5, view7)
MIN_STEPS = 50                           # consecutive visible frame pairs a track needs
BANDS = [('slow', 0.00, 0.25, 128), ('dynamic', 0.25, 0.75, 256), ('fast', 0.75, 0.95, 128)]
OUT_NAME = 'pose3d_512.npz'
CHUNK = 8000                             # tracks per chunk, to keep the memory of the distance computation low


def track_speeds(pose, vis, views):
    """pose (T, N, 3), vis (T, N, C) -> mean distance per visible frame pair (N,), and the number of pairs (N,)."""
    T, N, _ = pose.shape
    speed, steps = np.zeros(N), np.zeros(N, dtype=int)
    for s in range(0, N, CHUNK):
        x = pose[:, s:s + CHUNK]
        v = vis[:, s:s + CHUNK][..., views].any(-1) & np.isfinite(x).all(-1)    # (T, n)
        both = v[:-1] & v[1:]
        d = np.linalg.norm(x[1:] - x[:-1], axis=-1)
        d = np.where(both & np.isfinite(d), d, 0)
        steps[s:s + CHUNK] = both.sum(0)
        speed[s:s + CHUNK] = d.sum(0) / np.maximum(steps[s:s + CHUNK], 1)
    return speed, steps


def curate(trial_dir, trial_name):
    full = np.load(os.path.join(trial_dir, 'pose3d_full.npz'))
    pose, vis, keypoints = full['pose'][0], full['vis'][0], full['keypoints']        # (T, N, 3), (T, N, C), (N,)
    speed, steps = track_speeds(pose, vis, VIEWS)
    eligible = np.where(steps >= MIN_STEPS)[0]

    edges = np.quantile(speed[eligible], [lo for _, lo, _, _ in BANDS] + [BANDS[-1][2]])
    rng = np.random.default_rng([SEED, zlib.crc32(trial_name.encode())])
    chosen, band = [], []
    for b, (name, lo, hi, n) in enumerate(BANDS):
        last = b == len(BANDS) - 1
        in_band = eligible[(speed[eligible] >= edges[b]) & ((speed[eligible] <= edges[b + 1]) if last
                                                            else (speed[eligible] < edges[b + 1]))]
        if len(in_band) < n:
            raise ValueError(f'{trial_name}: only {len(in_band)} tracks in the {name} band, need {n}')
        chosen.append(rng.choice(in_band, n, replace=False))
        band.append(np.full(n, b))
    chosen, band = np.concatenate(chosen), np.concatenate(band)
    order = np.argsort(chosen)
    chosen, band = chosen[order], band[order]

    out = dict(
        pose=pose[:, chosen][None],
        vis=vis[:, chosen][None],
        keypoints=keypoints[chosen],
        band=band.astype(np.int8),
        speed=speed[chosen].astype(np.float32),
        source_index=chosen.astype(np.int64),
    )
    info = dict(n_total=pose.shape[1], n_eligible=len(eligible), edges=edges)
    return out, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', default=DATASET_ROOT)
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--split', default='test')
    ap.add_argument('--overwrite', action='store_true', help=f'replace an existing {OUT_NAME}')
    ap.add_argument('--dry-run', action='store_true', help='print the speed edges and counts, write nothing')
    args = ap.parse_args()

    for ds in args.datasets:
        split_dir = os.path.join(args.dataset_root, ds, args.split)
        for session in sorted(os.listdir(split_dir)):
            trial_dir = os.path.join(split_dir, session, 'trial')
            if not os.path.exists(os.path.join(trial_dir, 'pose3d_full.npz')):
                print(f'[skip] {ds}/{session}: no pose3d_full.npz')
                continue
            out_path = os.path.join(trial_dir, OUT_NAME)
            if os.path.exists(out_path) and not args.overwrite and not args.dry_run:
                print(f'[skip] {ds}/{session}: {OUT_NAME} exists (--overwrite to redo)')
                continue
            out, info = curate(trial_dir, f'{ds}/{session}')
            e = info['edges']
            print(f'{ds}/{session}: {info["n_eligible"]}/{info["n_total"]} eligible, speed edges '
                  f'{e[0]:.4f} | {e[1]:.4f} | {e[2]:.4f} | {e[3]:.4f}')
            if args.dry_run:
                continue
            tmp = out_path + '.tmp.npz'
            np.savez(tmp, **out)
            os.replace(tmp, out_path)


if __name__ == '__main__':
    main()
