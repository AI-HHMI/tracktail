#!/usr/bin/env python
"""Curates 512 tracks per trial and number of cameras for the dex_ycb n-camera ablation, shared by all three methods.

For N = 1 ... 8 the cameras are the nested sets view_00 ... view_{N-1} (N = 4 is the canonical set of the MVTracker
tables). For each N this samples 512 tracks of <trial>/pose3d.npz into <trial>/pose3d_ncam{N}_512.npz, in the format of
pose3d_cached.npz (pose3d.npz is never touched), so every pipeline can run on it with
--pose-file pose3d_ncam{N}_512.npz --cam-names <the N cameras> --query-first.

Tracks and query frames follow MVTracker's DexYCB loader (mvtracker/datasets/dexycb_multiview_dataset.py), without its
"hectic visibility" filter (a track that three or more cameras lose in the same frame; it removes under 3% of the tracks):
a track is eligible when it is visible in at least 2 frames in any of the N cameras. 512 eligible tracks are drawn at
random; a quarter of them are queried at a random visible frame (never the last visible one) and the others at their
first visible frame.

pose3d_ncam{N}_512.npz holds
    pose (1, T, n, 3), vis (1, T, n, C)   the sampled tracks, vis of all C cameras of pose3d.npz. Every frame before a
                                          track's query frame is blanked (vis False, pose nan): the first frame a track
                                          is valid in is then exactly its query frame, which is what query_first uses.
                                          The tracks are in the order of pose3d.npz
    keypoints (n,), query_times (n,)      the names and the query frames
    source_index (n,)                     the index of the track in pose3d.npz
n is 512, or all eligible tracks when there are fewer. The tracks are drawn without replacement with a generator seeded
by SEED, N and the trial name, so a rerun gives the same tracks.

Usage:
    python scripts/make_pose3d_ncam.py
    python scripts/make_pose3d_ncam.py --n-cams 2 4 --overwrite
"""
import argparse
import os
import zlib

import numpy as np

DATASET_ROOT = '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6'
DATASET = 'dex_ycb'

SEED = 0
N_TRACKS = 512
N_CAMS = list(range(1, 9))


def curate(trial_dir, trial_name, n_cams):
    data = np.load(os.path.join(trial_dir, 'pose3d.npz'))
    pose, vis, keypoints = data['pose'][0], data['vis'][0], data['keypoints']      # (T, N, 3), (T, N, C), (N,)
    T = pose.shape[0]
    vis = vis & np.isfinite(pose).all(-1)[..., None]

    seen = vis[..., :n_cams].any(-1)                                               # (T, N) seen in any of the cameras
    eligible = np.where(seen.sum(0) >= 2)[0]

    rng = np.random.default_rng([SEED, n_cams, zlib.crc32(trial_name.encode())])
    drawn = rng.permutation(eligible)[:N_TRACKS]                                   # random order
    n = len(drawn)

    # query frames as in MVTracker: the last visible frame is no query candidate; the first quarter of the drawn tracks
    # get a random visible frame, the others their first visible frame
    can_query = seen[:, drawn].copy()
    last_visible = (np.arange(T)[:, None] * can_query).max(0)
    can_query[last_visible, np.arange(n)] = False
    n_random = n // 4
    qt = np.empty(n, dtype=np.int64)
    qt[n_random:] = np.argmax(can_query[:, n_random:], axis=0)
    for i in range(n_random):
        qt[i] = rng.choice(np.flatnonzero(can_query[:, i]))

    order = np.argsort(drawn)
    chosen, qt = drawn[order], qt[order]

    after = np.arange(T)[:, None] >= qt[None, :]                                   # (T, n) from the query frame on
    out = dict(
        pose=np.where(after[..., None], pose[:, chosen], np.nan)[None],
        vis=(vis[:, chosen] & after[..., None])[None],
        keypoints=keypoints[chosen],
        query_times=qt.astype(np.int64),
        source_index=chosen.astype(np.int64),
    )
    return out, len(eligible)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', default=DATASET_ROOT)
    ap.add_argument('--split', default='test')
    ap.add_argument('--n-cams', nargs='+', type=int, default=N_CAMS)
    ap.add_argument('--overwrite', action='store_true', help='replace existing files')
    args = ap.parse_args()

    split_dir = os.path.join(args.dataset_root, DATASET, args.split)
    for n_cams in args.n_cams:
        name = f'pose3d_ncam{n_cams}_512.npz'
        sizes = []
        for session in sorted(os.listdir(split_dir)):
            trial_dir = os.path.join(split_dir, session, 'trial')
            out_path = os.path.join(trial_dir, name)
            if os.path.exists(out_path) and not args.overwrite:
                print(f'[skip] {session}: {name} exists (--overwrite to redo)')
                continue
            out, n_eligible = curate(trial_dir, f'{DATASET}/{session}', n_cams)
            sizes.append((n_eligible, len(out['keypoints'])))
            tmp = out_path + '.tmp.npz'
            np.savez(tmp, **out)
            os.replace(tmp, out_path)
        if sizes:
            e, n = zip(*sizes)
            print(f'N={n_cams}: {len(sizes)} trials, eligible tracks min {min(e)} / mean {int(np.mean(e))}, '
                  f'tracks written min {min(n)} / max {max(n)}')


if __name__ == '__main__':
    main()
