#!/usr/bin/env python
"""Samples 100 of the tracks of every pose3d_cached.npz at random, into pose3d_cached_100.npz next to it.

pose3d_cached.npz (scripts/make_pose3d_cached.py) holds the tracks MVTracker evaluates on (512 for kubric, 384 for
dex_ycb and cmupanoptic_3dgs). This keeps a random subset of 100 of them, so the 100-track runs are a subset of the
tracks of the full runs, in the same format (pose3d_cached.npz is never touched):

    pose (1, T, 100, 3), vis (1, T, 100, C)   the sampled tracks, in the order of the cached file. The frames before a
                                              track's query frame stay blanked, as in pose3d_cached.npz
    keypoints (100,)                          the names of the sampled tracks in pose3d_cached.npz
    query_times (100,)                        the frame each track is queried at
    cache_track_indices (100,)                the index of each track in MVTracker's cache file
    source_index (100,)                       the position of each track in pose3d_cached.npz

The tracks are drawn without replacement with a generator seeded by SEED and the dataset / trial name, so a rerun gives
the same tracks. Run the pipelines on it with --pose-file pose3d_cached_100.npz --query-first.

Usage:
    python scripts/make_pose3d_cached_100.py                       # all datasets, all trials
    python scripts/make_pose3d_cached_100.py --datasets dex_ycb --overwrite
"""
import argparse
import os
import zlib

import numpy as np

DATASET_ROOT = '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6'
DATASETS = ['kubric-multiview', 'dex_ycb', 'cmupanoptic_3dgs']

SEED = 0
N_TRACKS = 100
IN_NAME = 'pose3d_cached.npz'
OUT_NAME = 'pose3d_cached_100.npz'


def sample_trial(trial_dir, trial_name):
    cached = np.load(os.path.join(trial_dir, IN_NAME))
    n = cached['pose'].shape[2]
    if n < N_TRACKS:
        raise ValueError(f'{trial_name}: only {n} tracks, need {N_TRACKS}')
    rng = np.random.default_rng([SEED, zlib.crc32(trial_name.encode())])
    chosen = np.sort(rng.choice(n, N_TRACKS, replace=False))
    return dict(
        pose=cached['pose'][:, :, chosen],
        vis=cached['vis'][:, :, chosen],
        keypoints=cached['keypoints'][chosen],
        query_times=cached['query_times'][chosen],
        cache_track_indices=cached['cache_track_indices'][chosen],
        source_index=chosen.astype(np.int64),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', default=DATASET_ROOT)
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--split', default='test')
    ap.add_argument('--overwrite', action='store_true', help=f'replace an existing {OUT_NAME}')
    args = ap.parse_args()

    for ds in args.datasets:
        split_dir = os.path.join(args.dataset_root, ds, args.split)
        n_done = 0
        for session in sorted(os.listdir(split_dir)):
            trial_dir = os.path.join(split_dir, session, 'trial')
            if not os.path.exists(os.path.join(trial_dir, IN_NAME)):
                print(f'[skip] {ds}/{session}: no {IN_NAME}')
                continue
            out_path = os.path.join(trial_dir, OUT_NAME)
            if os.path.exists(out_path) and not args.overwrite:
                print(f'[skip] {ds}/{session}: {OUT_NAME} exists (--overwrite to redo)')
                continue
            out = sample_trial(trial_dir, f'{ds}/{session}')
            tmp = out_path + '.tmp.npz'
            np.savez(tmp, **out)
            os.replace(tmp, out_path)
            n_done += 1
        print(f'{ds}: wrote {n_done} files')


if __name__ == '__main__':
    main()
