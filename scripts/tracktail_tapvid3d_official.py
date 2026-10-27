#!/usr/bin/env python
"""Tracktail on TAPVid-3D with the official benchmark protocol, for a like-for-like comparison with ARROW / D4RT.

Differences from eval_testset_metrics.py (which queries each track at its first visible frame and tracks forward only):
  * each track is queried at the benchmark's own query (queries_xyt: pixel + frame t_q), with its ground-truth 3D point
    at t_q as tracktail's 3D query (tracktail queries are 3D: this gives it the query's metric depth, which the RGB-only
    methods do not get; the per-video median scaling of the official metric removes the global scale advantage);
  * it is tracked BOTH ways: forward from t_q on the video, and backward from t_q by running the same windowed tracker
    on the time-reversed video (frames symlinked in reverse order); the two halves are joined at t_q, so every frame
    of every track has a prediction, as the official metric expects;
  * predictions are saved in the eval_testset_metrics.py npz format (camera coordinates; the tapvid3d_* trials have
    identity extrinsics) and scored with scripts/tapvid3d_official_metrics.py (all frames, per-video median scaling).

Predictions already in <out>/<dataset>/ are reused. Only static-camera metadata is supported (the tapvid3d_* trials).

--max-frames 48 gives the short-clip setting (the first 48 frames, queries in them only), which D4RT reportedly used
and ARROW also reports (github.com/ifradlin/arrow/issues/3).

Usage:
    pixi run python scripts/tracktail_tapvid3d_official.py --wandb-folder <run dir> --out <dir> \\
        [--datasets tapvid3d_pstudio ...] [--trials-json arrow_baseline/tapvid3d_minival.json | --all-trials]
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tracktail.model.eval_metrics import TRACKTAIL_VIS_THRESH       # noqa: E402
from tracktail.inference.inference_utils import (                 # noqa: E402
    load_model_from_base_folder, load_camera_group_from_metadata, run_tracker_encoder_on_videos)

DEFAULT_ROOT = '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6'
DEFAULT_TRIALS = os.path.join(REPO, 'arrow_baseline', 'tapvid3d_minival.json')
DATASETS = ['tapvid3d_pstudio', 'tapvid3d_drivetrack', 'tapvid3d_adt']


def reversed_frames_dir(img_dir, n_frames, tmp_root):
    """A folder whose i-th image (sorted name) is frame n_frames-1-i of img_dir."""
    files = sorted(os.listdir(img_dir))[:n_frames]
    ext = os.path.splitext(files[0])[1]
    d = tempfile.mkdtemp(dir=tmp_root)
    for i, f in enumerate(reversed(files)):
        os.symlink(os.path.join(img_dir, f), os.path.join(d, f'{i:06d}{ext}'))
    return d


RAW_TAPVID3D = '/groups/karashchuk/karashchuklab/animal-datasets/objects/tapvid3d'
ADT_POSES = '/groups/karashchuk/home/karashchukl/results/paper-final-v2/tapvid3d_adt_minival_camera_poses'


def cam_to_first(ds, tid, T, adt_poses):
    """(T, 4, 4) transforms camera t -> camera 0 from the ground-truth poses: DriveTrack's official extrinsics_w2c,
    ADT's poses recovered from the raw sequences (adt_poses/<tid>.npz c2w)."""
    if ds == 'tapvid3d_drivetrack':
        z = np.load(os.path.join(RAW_TAPVID3D, 'drivetrack', f'{tid}.npz'), allow_pickle=True)
        c2w = np.linalg.inv(z['extrinsics_w2c'].astype(np.float64))
    elif ds == 'tapvid3d_adt':
        c2w = np.load(os.path.join(adt_poses, f'{tid}.npz'))['c2w'].astype(np.float64)
    else:
        raise ValueError(f'no ground-truth camera poses for {ds} (static camera)')
    if len(c2w) < T:
        raise ValueError(f'{len(c2w)} poses for {T} frames')
    return np.linalg.inv(c2w[0]) @ c2w[:T]


def moving_camera_group(meta, w2c, tmp_root):
    """Camera group with per-frame extrinsics w2c (T, 4, 4), through a temporary moving_cams metadata file."""
    m = dict(meta)
    cam = next(iter(m['intrinsic_matrices']))
    m['moving_cams'] = True
    m['extrinsic_matrices'] = {cam: np.asarray(w2c, dtype=np.float64).tolist()}
    fd, path = tempfile.mkstemp(suffix='.yaml', dir=tmp_root)
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(m, f)
    return load_camera_group_from_metadata(path, device='cpu')


def track_one_way(model, video_dir, camera_group, q3d, qt, occ, T, args, device):
    out = run_tracker_encoder_on_videos(
        model=model, video_paths=[video_dir], camera_group=camera_group,
        query_points_3d=torch.as_tensor(q3d, dtype=torch.float32), start_frame=0, n_frames=T,
        n_overlap=args.n_overlap, max_kpts=args.max_kpts, device=device, occlusion_gt=occ,
        query_times=torch.as_tensor(qt, dtype=torch.int32), motion_margin=True, vis_thresh=args.vis_thresh)
    fn = out['frame_numbers'].numpy()
    if len(fn) != T or not np.array_equal(fn, np.arange(T)):
        raise RuntimeError(f'tracker returned frames {fn[:3]}..{fn[-1:]} for T={T}')
    cp = out['coords_pred'][0].float().numpy()                   # (T, N, 3)
    vp = out['vis_pred'][0].float().numpy()                      # (T, N, 1)
    vp2d = out['vis_pred_2d'][:, 0].float().numpy() if 'vis_pred_2d' in out else None   # (cams, T, N)
    return cp, vp, vp2d


def process_trial(model, trial_path, npz_path, args, device, tmp_root):
    with open(os.path.join(trial_path, 'metadata.yaml')) as f:
        meta = yaml.safe_load(f)
    if meta.get('moving_cams', False) or len(meta['intrinsic_matrices']) != 1:
        raise ValueError('only single static-camera trials are supported')
    cam = next(iter(meta['intrinsic_matrices']))
    img_dir = os.path.join(trial_path, 'img', cam)
    camera_group = load_camera_group_from_metadata(os.path.join(trial_path, 'metadata.yaml'), device='cpu')

    pose = np.load(os.path.join(trial_path, 'pose3d.npz'))
    coords = pose['pose'][0].astype(np.float64)                  # (T, N, 3) camera coordinates, NaN when occluded
    vis = pose['vis'][0, ..., 0].astype(bool)                    # (T, N)
    xyt = pose['queries_xyt'].astype(np.float64)                 # (N, 3) pixel x, y, frame
    T = min(coords.shape[0], len(os.listdir(img_dir)))
    if args.max_frames:
        # D4RT-style short clips: the first max_frames frames, with the queries whose frame falls inside them
        T = min(T, args.max_frames)
        keep = np.round(xyt[:, 2]).astype(np.int64) < T
        coords, vis, xyt = coords[:, keep], vis[:, keep], xyt[keep]
        if not keep.any():
            raise ValueError(f'no query in the first {T} frames')
    coords, vis = coords[:T], vis[:T]
    N = coords.shape[1]
    qt = np.clip(np.round(xyt[:, 2]).astype(np.int64), 0, T - 1)
    q3d = coords[qt, np.arange(N)]
    if not np.isfinite(q3d).all():
        raise ValueError(f'{(~np.isfinite(q3d).all(-1)).sum()} queries have no finite ground truth at their frame')
    occ = torch.as_tensor(vis[qt, np.arange(N)].astype(np.int64)[:, None])      # (N, 1): visible at the query

    cg_f = cg_b = camera_group
    q_in = q3d
    if args.gt_poses:
        # calibrated moving camera: world = the first camera; queries and predictions in that frame
        to0 = cam_to_first(args.current_ds, args.current_tid, T, args.adt_poses)        # camera t -> camera 0
        w2c = np.linalg.inv(to0)                                                        # camera 0 -> camera t
        cg_f = moving_camera_group(meta, w2c, tmp_root)
        cg_b = moving_camera_group(meta, w2c[::-1], tmp_root)
        q_in = np.einsum('nij,nj->ni', to0[qt, :3, :3], q3d) + to0[qt, :3, 3]
    cp_f, vp_f, v2_f = track_one_way(model, img_dir, cg_f, q_in, qt, occ, T, args, device)
    rev_dir = reversed_frames_dir(img_dir, T, tmp_root)
    try:
        cp_b, vp_b, v2_b = track_one_way(model, rev_dir, cg_b, q_in, T - 1 - qt, occ, T, args, device)
    finally:
        shutil.rmtree(rev_dir, ignore_errors=True)
    cp_b, vp_b = cp_b[::-1], vp_b[::-1]                          # back to forward time
    fwd = (np.arange(T)[:, None] >= qt[None])                    # (T, N): forward half (incl. the query frame)
    cp = np.where(fwd[..., None], cp_f, cp_b)
    vp = np.where(fwd[..., None], vp_f, vp_b)
    cp_world = None
    if args.gt_poses:
        # back to each frame's camera coordinates, for the (camera-coordinate) official metric
        cp_world = cp
        cp = np.einsum('tij,tnj->tni', w2c[:, :3, :3], cp_world) + w2c[:, None, :3, 3]
    save = dict(
        coords_pred=cp[None].astype(np.float32), vis_pred=vp[None].astype(np.float32),
        coords_true=coords[None].astype(np.float32), vis_true=vis[None, ..., None],
        vis_true_cams=vis[None, ..., None], query_times=qt.astype(np.int32),
        query_xyt=xyt.astype(np.float32), frame_numbers=np.arange(T),
        cam_names_used=np.array([cam]), cam_indices_used=np.array([0], dtype=np.int32),
        protocol=np.array('tapvid3d queries_xyt, bidirectional (forward + time-reversed)'
                          + (', ground-truth camera poses (world = first camera)' if args.gt_poses else '')))
    if cp_world is not None:
        save['coords_pred_world'] = cp_world[None].astype(np.float32)
    if v2_f is not None and v2_b is not None:
        save['vis_pred_2d'] = np.where(fwd[None], v2_f, v2_b[:, ::-1])[:, None].astype(np.float32)
    np.savez(npz_path, **save)
    return T, N


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wandb-folder', required=True)
    ap.add_argument('--checkpoint', type=int, default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--trials-json', default=DEFAULT_TRIALS, help='{dataset: [<seq>_<id>, ...]} (default: mini-val)')
    ap.add_argument('--all-trials', action='store_true', help='every trial of <root>/<dataset>/test instead')
    ap.add_argument('--max-frames', type=int, default=None,
                    help='evaluate only the first N frames (D4RT reportedly uses 48), keeping the queries inside them')
    ap.add_argument('--gt-poses', action='store_true',
                    help='give tracktail the ground-truth per-frame camera poses (moving calibrated camera, world = the '
                         'first camera; DriveTrack and ADT only); predictions are saved in camera coordinates as usual')
    ap.add_argument('--adt-poses', default=ADT_POSES, help='ADT camera poses recovered from the raw sequences')
    ap.add_argument('--vis-thresh', type=float, default=TRACKTAIL_VIS_THRESH,
                    help='visibility threshold of the predicted occlusion carried between windows (occlusion embedding); '
                         'score with the same --vis-thresh in tapvid3d_official_metrics.py')
    ap.add_argument('--n-overlap', type=int, default=8)
    ap.add_argument('--max-kpts', type=int, default=2500, help='keypoint chunk size (as eval_testset_metrics.py)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    trials = {}
    if not args.all_trials:
        with open(args.trials_json) as f:
            listed = json.load(f)
    for ds in args.datasets:
        base = os.path.join(args.root, ds, 'test')
        if args.all_trials:
            ids = [f'{s}_{t}' for s in sorted(os.listdir(base)) for t in sorted(os.listdir(os.path.join(base, s)))]
        else:
            ids = listed[ds]
        trials[ds] = [(tid, os.path.join(base, *tid.rsplit('_', 1))) for tid in ids]
        trials[ds] = [(tid, p) for tid, p in trials[ds] if os.path.isdir(p)]

    device = torch.device(args.device)
    model = None
    tmp_root = tempfile.mkdtemp(prefix='tracktail_tapvid3d_rev_')
    t_start = time.time()
    try:
        for ds in args.datasets:
            os.makedirs(os.path.join(args.out, ds), exist_ok=True)
            for i, (tid, path) in enumerate(trials[ds]):
                npz = os.path.join(args.out, ds, f'{tid}.npz')
                if os.path.exists(npz) and not args.force:
                    continue
                if model is None:
                    model = load_model_from_base_folder(args.wandb_folder, checkpoint=args.checkpoint,
                                                        device=device)[0]
                t0 = time.time()
                args.current_ds, args.current_tid = ds, tid
                try:
                    T, N = process_trial(model, path, npz, args, device, tmp_root)
                    info = f'T={T} N={N}'
                except Exception as exc:                          # noqa: BLE001
                    info = f'FAILED: {str(exc).splitlines()[0][:120]}'
                torch.cuda.empty_cache()
                print(f'[{ds} {i + 1}/{len(trials[ds])}] {tid}: {info} ({time.time() - t0:.0f}s, '
                      f'elapsed {time.time() - t_start:.0f}s)', flush=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print('done')


if __name__ == '__main__':
    main()
