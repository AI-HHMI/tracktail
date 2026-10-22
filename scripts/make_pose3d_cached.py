#!/usr/bin/env python
"""Converts MVTracker's cached track files into tracktail pose files, one per trial.

MVTracker evaluates on a fixed set of tracks per sequence (<native trial>/cache/cachedtracks--*.npz, shipped with
its datasets): the tracks, the frame each track is queried at, and its visibility in the 4 cameras of the paper
tables. This writes the same tracks as <trial>/pose3d_cached.npz next to the trial's pose3d.npz (which is never
touched), so any tracktail pipeline can run on them with --pose-file pose3d_cached.npz.

pose3d_cached.npz holds
    pose (1, T, N, 3)   the cached tracks, in the world frame of pose3d.npz. N is the cache's N, repeated entries included
    vis  (1, T, N, C)   visibility in the 4 cameras of the cache, placed at their positions in the camera list of
                        metadata.yaml (C columns, all other cameras False). Positions in that list are what the
                        inference code indexes with, so panoptic gets 27 columns here, not the 31 of pose3d.npz
    keypoints (N,)      kpt0, kpt1, ...
    query_times (N,)    the frame each track is queried at (and, for provenance, cache_track_indices (N,))
Every frame before a track's query frame is blanked (vis False, pose nan): the first frame a track is valid in
is then exactly its query frame, which is what query_first uses, including for the tracks MVTracker queries at a
random visible frame instead of the first one.

Usage:
    python scripts/make_pose3d_cached.py                      # all datasets, all trials
    python scripts/make_pose3d_cached.py --dry-run
    python scripts/make_pose3d_cached.py --datasets dex_ycb --check
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

DATASET_ROOT = '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6'
NATIVE_ROOT = '/groups/karashchuk/karashchuklab/mvtracker-datasets'

# the cameras of the MVTracker tables: names in metadata.yaml, and the cache file of that view set
DATASETS = {
    'kubric-multiview': dict(native_dir = 'kubric-multiview/test', cams = ['view_0', 'view_1', 'view_2', 'view_3'],
                             cache = 'cachedtracks--seed72-dynamic0.5-verydynamic-0.25-views0_1_2_3-n512-numviews-1-t24--v1.npz'),
    'dex_ycb': dict(native_dir = 'dex-ycb-multiview', cams = ['view_00', 'view_01', 'view_02', 'view_03'],
                    cache = 'cachedtracks--seed72-views0_1_2_3-n384--v1.npz'),
    'cmupanoptic_3dgs': dict(native_dir = 'panoptic-multiview', cams = ['1', '7', '14', '20'],
                             cache = 'cachedtracks--seed72-views1_7_14_20-n384--v1.npz'),
}


def get_camera_names(metadata_path):
    '''camera names in the order the inference code uses (numeric for numeric names, else alphabetical)'''

    with open(metadata_path) as f:
        names = list(yaml.safe_load(f)['intrinsic_matrices'].keys())

    if all(n.isdigit() for n in names):
        return sorted(names, key = int)
    return sorted(names)


def get_cached_pose(cache_path, camera_names, cams):
    '''
    builds pose and vis of one trial from its cache file

    parameters:
        cache_path: cachedtracks--*.npz of the trial
        camera_names: all camera names of the trial's metadata.yaml, in order
        cams: the cameras of the cache, in the order of its views

    returns:
        dict with the entries of pose3d_cached.npz
    '''

    cache = np.load(cache_path)
    coords = cache['traj3d_world'].astype(np.float64)               # T, N, 3
    vis_views = cache['visibility'].astype(bool)                    # V, T, N
    query_times = cache['query_points'][:, 0].round().astype(int)   # N
    T, N, _ = coords.shape

    if vis_views.shape[0] != len(cams):
        raise ValueError(f'{cache_path}: cache has {vis_views.shape[0]} views, expected {len(cams)} ({cams})')

    positions = [camera_names.index(c) for c in cams]
    vis = np.zeros((T, N, len(camera_names)), dtype = bool)
    vis[:, :, positions] = vis_views.transpose(1, 2, 0)

    # every track has to be visible at its query frame
    if not vis[query_times, np.arange(N)].any(axis = -1).all():
        raise ValueError(f'{cache_path}: some tracks are not visible at their query frame')

    # blank all frames before the query frame, so that it is the first valid frame of the track
    before = np.arange(T)[:, None] < query_times[None, :]           # T, N
    vis[before] = False
    coords[before] = np.nan

    return dict(pose = coords[None], vis = vis[None],
                keypoints = np.array([f'kpt{i}' for i in range(N)]),
                query_times = query_times,
                cache_track_indices = cache['track_indices'])


def check_trial(trial_path, pose_file, cams, expected):
    '''loads the written file with the tracktail loader and compares tracks and query frames to the cache'''

    from tracktail.inference.inference_utils import load_trial

    trial = load_trial(trial_path, pose_file = pose_file)
    n = len(expected['query_times'])
    ok = (int(trial['valid_flat'].sum()) == n
          and np.array_equal(np.asarray(trial['query_times_flat']), expected['query_times'])
          and np.array_equal(np.asarray(trial['vis_gt'])[0].shape[:2], expected['pose'][0].shape[:2]))
    return ok


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', default = DATASET_ROOT, help = 'tracktail-format datasets (where pose3d_cached.npz is written)')
    ap.add_argument('--native-root', default = NATIVE_ROOT, help = 'MVTracker datasets (where the cache files are)')
    ap.add_argument('--datasets', nargs = '+', default = list(DATASETS))
    ap.add_argument('--split', default = 'test')
    ap.add_argument('--name', default = 'pose3d_cached.npz', help = 'file written in every trial folder')
    ap.add_argument('--overwrite', action = 'store_true', help = 'replace an existing --name file')
    ap.add_argument('--dry-run', action = 'store_true', help = 'only list what would be written')
    ap.add_argument('--check', action = 'store_true', help = 'reload every written file with load_trial and compare')
    args = ap.parse_args()

    if args.name == 'pose3d.npz':
        raise SystemExit('refusing to write pose3d.npz: choose another --name')

    n_written = n_skipped = n_failed = n_bad = 0
    for ds in args.datasets:
        cfg = DATASETS[ds]
        split_path = os.path.join(args.dataset_root, ds, args.split)
        for session in sorted(os.listdir(split_path)):
            trial_path = os.path.join(split_path, session, 'trial')
            cache_path = os.path.join(args.native_root, cfg['native_dir'], session, 'cache', cfg['cache'])
            out_path = os.path.join(trial_path, args.name)
            if not os.path.isdir(trial_path) or not os.path.exists(cache_path):
                print(f'{ds}/{session}: no trial folder or no cache file ({cache_path}), skipped')
                n_failed += 1
                continue
            if os.path.exists(out_path) and not args.overwrite:
                print(f'{ds}/{session}: {args.name} exists, skipped (--overwrite to replace)')
                n_skipped += 1
                continue

            try:
                names = get_camera_names(os.path.join(trial_path, 'metadata.yaml'))
                entries = get_cached_pose(cache_path, names, cfg['cams'])
            except Exception as e:
                print(f'{ds}/{session}: FAILED {e}')
                n_failed += 1
                continue

            N = len(entries['query_times'])
            print(f'{ds}/{session}: {N} tracks, {entries["pose"].shape[1]} frames, {entries["vis"].shape[-1]} camera columns'
                  + (' (dry run)' if args.dry_run else ''))
            if args.dry_run:
                continue

            np.savez(out_path, **entries)
            n_written += 1
            if args.check and not check_trial(trial_path, args.name, cfg['cams'], entries):
                print(f'    CHECK FAILED for {out_path}')
                n_bad += 1

    print(f'\nwritten {n_written}, skipped {n_skipped}, failed {n_failed}' + (f', check failed {n_bad}' if args.check else ''))


if __name__ == '__main__':
    main()
