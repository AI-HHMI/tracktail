#!/usr/bin/env python
"""Official TAPVid-3D metrics (AJ, APD, OA) for predictions saved in the eval_testset_metrics.py / arrow_inference.py
npz format (<out>/<dataset>/<seq>_<id>.npz with coords_pred, coords_true, vis_pred, vis_true in camera coordinates).

This is the scoring of the TAPVid-3D benchmark (tapnet/tapvid3d/evaluation/metrics.py, compute_tapvid3d_metrics,
vendored below without the einops dependency), which the ARROW / D4RT papers report:
  * depth-adaptive thresholds: a visible point is within threshold d in {1, 2, 4, 8, 16} when its 3D error is below
    d * z_gt / sqrt(fx * fy) (d pixels back-projected at the point's ground-truth depth), with the intrinsics rescaled
    to a smallest image side of 256 px (evaluate_model.py's metric_eval_resolution);
  * predictions rescaled per video by median ||x_gt|| / median ||x_pred|| over the points visible in both ('median',
    the default; 'none' keeps the saved scale);
  * every frame of every track is scored (both before and after the query frame), and metrics are pooled over the
    track-frames of a video, then averaged over videos.
This differs from the depth-relative *_cam metrics of eval_testset_metrics.py (thresholds of 0.01 ... 2.56 x depth,
per-track averaging, pre-query frames excluded), which are much more lenient.

Writes <out>/tapvid3d_official.json (per dataset: mean over trials + per-trial values) and tapvid3d_official.md,
with the ARROW-paper numbers (arXiv 2610.01314, Table 2, camera coordinates, mini-val) alongside.

Usage:
    pixi run python scripts/tapvid3d_official_metrics.py --out <preds_dir> [--datasets tapvid3d_adt ...]
        [--root <dataset root>] [--trials-json arrow_baseline/tapvid3d_minival.json] [--scaling median]
"""
import argparse
import glob
import json
import os

import numpy as np
import yaml

DEFAULT_ROOT = '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6'
DATASETS = ['tapvid3d_adt', 'tapvid3d_drivetrack', 'tapvid3d_pstudio']

# ARROW paper (arXiv 2610.01314v1), Table 2, camera-coordinate 3D tracking on the TAPVid-3D mini-val split,
# without ground-truth intrinsics: (AJ, APD, OA). Baselines in that table are from D4RT.
PAPER = {
    'ARROW': {'tapvid3d_drivetrack': (0.241, 0.350, 0.816), 'tapvid3d_adt': (0.360, 0.473, 0.902),
              'tapvid3d_pstudio': (0.141, 0.201, 0.870), 'avg': (0.248, 0.341, 0.863)},
    'D4RT': {'tapvid3d_drivetrack': (0.257, 0.345, 0.875), 'tapvid3d_adt': (0.240, 0.319, 0.926),
             'tapvid3d_pstudio': (0.138, 0.186, 0.897), 'avg': (0.212, 0.283, 0.899)},
}


# ── vendored from tapnet/tapvid3d/evaluation/metrics.py (Apache 2.0), 'n t' inputs, one video ──────────────

def compute_tapvid3d_metrics(gt_occluded, gt_tracks, pred_occluded, pred_tracks, intrinsics_params,
                             scaling='median', eval_mask=None):
    """gt_occluded, pred_occluded (n, t) bool; gt_tracks, pred_tracks (n, t, 3); intrinsics [fx, fy, cx, cy].
    eval_mask (n, t) bool (not in the official code, default all True): track-frames that are scored, as the
    evaluation weights of the official function (and excluded from the median scale)."""
    gt_occluded, pred_occluded = gt_occluded[None], pred_occluded[None]
    gt_tracks, pred_tracks = gt_tracks[None], pred_tracks[None]
    summing_axis = (-2, -1)
    evaluation_weights = np.ones(gt_occluded.shape) if eval_mask is None else eval_mask[None].astype(np.float64)
    metrics = {}

    pred_norms = np.sqrt(np.maximum(1e-12, np.sum(np.square(pred_tracks), axis=-1)))
    gt_norms = np.sqrt(np.maximum(1e-12, np.sum(np.square(gt_tracks), axis=-1)))
    if scaling == 'none':
        scale_factor = 1.0
    else:
        either_occluded = np.logical_or(gt_occluded, pred_occluded) | (evaluation_weights == 0)
        pred_norms = np.where(either_occluded, np.nan, pred_norms)
        gt_norms = np.where(either_occluded, np.nan, gt_norms)
        agg = np.nanmedian if scaling == 'median' else np.nanmean
        scale_factor = agg(gt_norms, axis=(-2, -1), keepdims=True) / agg(pred_norms, axis=(-2, -1), keepdims=True)
        if not np.all(np.isfinite(scale_factor)):           # nothing visible in both: keep the saved scale
            scale_factor = 1.0
    pred_tracks = pred_tracks * scale_factor

    metrics['occlusion_accuracy'] = np.sum(np.equal(pred_occluded, gt_occluded) * evaluation_weights,
                                           axis=summing_axis) / np.sum(evaluation_weights, axis=summing_axis)
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    mean_focal_length = np.sqrt(intrinsics_params[0] * intrinsics_params[1])
    multiplier = gt_tracks[..., -1] / mean_focal_length
    all_frac_within, all_jaccard = [], []
    for thresh in [1, 2, 4, 8, 16]:
        pointwise_thresh = thresh * multiplier
        within_dist = np.sum(np.square(pred_tracks - gt_tracks), axis=-1) < np.square(pointwise_thresh)
        is_correct = np.logical_and(within_dist, visible)
        count_correct = np.sum(is_correct * evaluation_weights, axis=summing_axis)
        count_visible_points = np.sum(visible * evaluation_weights, axis=summing_axis)
        frac_correct = count_correct / count_visible_points
        metrics['pts_within_' + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)
        true_positives = np.sum((is_correct & pred_visible) * evaluation_weights, axis=summing_axis)
        gt_positives = np.sum(visible * evaluation_weights, axis=summing_axis)
        false_positives = ((~visible) & pred_visible) | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives * evaluation_weights, axis=summing_axis)
        jaccard = true_positives / (gt_positives + false_positives)
        metrics['jaccard_' + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics['average_jaccard'] = np.mean(np.stack(all_jaccard, axis=-2), axis=-2)
    metrics['average_pts_within_thresh'] = np.mean(np.stack(all_frac_within, axis=-2), axis=-2)
    metrics['scale_factor'] = np.asarray(scale_factor).reshape(-1)[:1]
    return {k: float(np.asarray(v).reshape(-1)[0]) for k, v in metrics.items()}


# ── npz -> metric inputs ──────────────────────────────────────────────────────────────────────────────────────

def intrinsics_of(trial_path, eval_resolution=256):
    """[fx, fy, cx, cy] rescaled so that the image's smallest side is eval_resolution, as evaluate_model.py does
    (metric_eval_resolution=256, used for all TAPVid-3D paper results): the pixel thresholds are at 256 px."""
    with open(os.path.join(trial_path, 'metadata.yaml')) as f:
        meta = yaml.safe_load(f)
    cam = next(iter(meta['intrinsic_matrices']))
    K = np.asarray(meta['intrinsic_matrices'][cam], dtype=np.float64)
    hw = (meta['camera_heights'][cam], meta['camera_widths'][cam])
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]) * (eval_resolution / min(hw))


def score_npz(path, trial_path, scaling, post_query=False, vis_thresh=0.5):
    with np.load(path, allow_pickle=True) as d:
        cp = np.asarray(d['coords_pred'], dtype=np.float64)[0]      # (T, N, 3), camera coordinates
        ct = np.asarray(d['coords_true'], dtype=np.float64)[0]
        vp = np.asarray(d['vis_pred'], dtype=np.float64)[0, ..., 0]  # (T, N) logits
        if 'vis_true_cams' in d.files:
            vt = np.asarray(d['vis_true_cams'])[0].astype(np.float64)
            vt = (np.isfinite(vt) & (vt > 0.5)).any(-1)
        else:
            vt = np.asarray(d['vis_true']).astype(bool)[0, ..., 0]
        qt = np.asarray(d['query_times']) if 'query_times' in d.files else None
    vt = vt & np.isfinite(ct).all(-1)
    mask = None
    if post_query:
        # forward-only predictions (tracktail's query-first): score only the query frame and later
        if qt is None:
            raise ValueError('--post-query needs query_times in the predictions')
        mask = (np.arange(ct.shape[0])[:, None] >= qt[None]).T             # (n, t)
    pred_vis = (1 / (1 + np.exp(-vp))) > vis_thresh
    # occluded GT positions may be NaN (cleaned placeholders): they never count as within threshold
    ct = np.where(vt[..., None], ct, np.nan)
    cp = np.where(np.isfinite(cp), cp, 1e6)                         # missing predictions: far away
    m = compute_tapvid3d_metrics(~vt.T, ct.transpose(1, 0, 2), ~pred_vis.T, cp.transpose(1, 0, 2),
                                 intrinsics_of(trial_path), scaling=scaling, eval_mask=mask)
    m['n_tracks'] = int(ct.shape[1])
    m['n_frames'] = int(ct.shape[0])
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='predictions folder: <out>/<dataset>/<seq>_<id>.npz')
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--datasets', nargs='+', default=None, help='default: the tapvid3d_* folders in --out')
    ap.add_argument('--trials-json', default=None,
                    help='json {dataset: [trial ids]} restricting the scored trials (e.g. tapvid3d_minival.json)')
    ap.add_argument('--scaling', default='median', choices=['median', 'mean', 'none'])
    ap.add_argument('--vis-thresh', type=float, default=0.5,
                    help='predicted visible when sigmoid(vis logit) > this (official: 0.5); affects AJ, OA and which '
                         'points enter the median scale')
    ap.add_argument('--post-query', action='store_true',
                    help='score only frames at/after each track\'s query frame (forward-only trackers); not the '
                         'official protocol, which scores every frame')
    ap.add_argument('--tag', default='', help='suffix of the output files (tapvid3d_official<tag>.json/.md)')
    args = ap.parse_args()

    datasets = args.datasets or [ds for ds in DATASETS if os.path.isdir(os.path.join(args.out, ds))]
    subset = None
    if args.trials_json:
        with open(args.trials_json) as f:
            subset = {k: set(v) for k, v in json.load(f).items() if isinstance(v, list)}

    results = {}
    for ds in datasets:
        per_trial = {}
        for path in sorted(glob.glob(os.path.join(args.out, ds, '*.npz'))):
            tid = os.path.basename(path)[:-4]
            if subset is not None and tid not in subset.get(ds, ()):
                continue
            seq, trial = tid.rsplit('_', 1)
            trial_path = os.path.join(args.root, ds, 'test', seq, trial)
            try:
                per_trial[tid] = score_npz(path, trial_path, args.scaling, args.post_query, args.vis_thresh)
            except Exception as exc:                                  # noqa: BLE001
                print(f'  [fail] {ds}/{tid}: {exc}')
        if not per_trial:
            print(f'{ds}: no predictions')
            continue
        keys = [k for k in next(iter(per_trial.values())) if k not in ('n_tracks', 'n_frames')]
        summary = {k: float(np.nanmean([m[k] for m in per_trial.values()])) for k in keys}
        summary['n_trials'] = len(per_trial)
        if subset is not None:
            summary['n_trials_expected'] = len(subset.get(ds, ()))
        results[ds] = {'summary': summary, 'per_trial': per_trial}
        print(f'{ds}: {len(per_trial)} trials  AJ={summary["average_jaccard"]:.3f}  '
              f'APD={summary["average_pts_within_thresh"]:.3f}  OA={summary["occlusion_accuracy"]:.3f}')

    if not results:
        return
    order = [ds for ds in ['tapvid3d_drivetrack', 'tapvid3d_adt', 'tapvid3d_pstudio'] if ds in results]
    if len(order) == 3:
        results['avg'] = {'summary': {k: float(np.mean([results[ds]['summary'][k] for ds in order]))
                                      for k in ('average_jaccard', 'average_pts_within_thresh', 'occlusion_accuracy')}}
    meta = dict(out=args.out, scaling=args.scaling, trials_json=args.trials_json, post_query=args.post_query,
                vis_thresh=args.vis_thresh)
    with open(os.path.join(args.out, f'tapvid3d_official{args.tag}.json'), 'w') as f:
        json.dump({'meta': meta, **results}, f, indent=1)

    rows = order + (['avg'] if 'avg' in results else [])
    L = ['# TAPVid-3D official metrics (camera coordinates)', '',
         f'compute_tapvid3d_metrics (depth-adaptive 1-16 px thresholds, scaling={args.scaling}, '
         f'{"frames at/after the query only" if args.post_query else "all frames"}, '
         f'visibility threshold {args.vis_thresh}), mean '
         f'over trials. Trials: {"all predictions" if subset is None else args.trials_json}. Paper: ARROW arXiv '
         '2610.01314 Table 2 (mini-val, no GT intrinsics).', '',
         '| dataset | #tr | AJ | APD | OA | paper ARROW AJ / APD / OA | paper D4RT AJ / APD / OA |',
         '|---|---|---|---|---|---|---|']
    for ds in rows:
        s = results[ds]['summary']
        paper = ' | '.join(' / '.join(f'{v:.3f}' for v in PAPER[m][ds]) for m in ('ARROW', 'D4RT'))
        L.append(f'| {ds} | {s.get("n_trials", "")} | {s["average_jaccard"]:.3f} | '
                 f'{s["average_pts_within_thresh"]:.3f} | {s["occlusion_accuracy"]:.3f} | {paper} |')
    with open(os.path.join(args.out, f'tapvid3d_official{args.tag}.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
