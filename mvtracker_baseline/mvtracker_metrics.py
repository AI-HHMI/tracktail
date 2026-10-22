"""
Compute eval metrics for MVTracker predictions.

Reads mvtracker_3d.npz from output_root, loads GT from dataset_root,
computes metrics with tracktail/model/eval_metrics_paper.py (the MVTracker paper's protocol, the default) or
with get_eval_metrics (--legacy-metrics), saves <trial>_metrics.npz alongside each prediction, and aggregates to CSV.

Identical logic to cotracker_metrics.py — only the prediction filename and
default metric prefix differ.

Structure expected:
    output_root/dataset/split/session/trial/mvtracker_3d.npz
    dataset_root/dataset/split/session/trial/pose3d.npz

Usage:
    python mvtracker_metrics.py \\
        --dataset-root /path/to/datasets \\
        --output-root  /path/to/mvtracker-outputs \\
        --datasets kubric-multiview dex_ycb cmupanoptic_3dgs \\
        --split test
"""

import os
import argparse
import traceback

import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

try:
    from tracktail.model.eval_metrics import get_eval_metrics
    from tracktail.model.eval_metrics_paper import PAPER_MOVEMENT_THRESHOLDS, get_eval_metrics_paper
except ImportError:   # checkout that still uses the old package name (no paper metrics there)
    from posetail.posetail.eval_metrics import get_eval_metrics


DATASET_THRESHOLDS = {
    'dex_ycb':                  [x / 100 for x in [1, 2, 5, 10, 20]],
    'kubric-multiview':         [x / 13 for x in [0.65, 1.3, 2.6, 5.2, 10.4]],
    'cmupanoptic_3dgs':         [x / 100 for x in [5, 10, 20, 40]],
    'point-odyssey-human':      [0.30, 0.60, 1.20, 2.40, 4.80],
    'po-animal-sub256-motion':  [0.30, 0.60, 1.20, 2.40, 4.80],
}

# failure distance of the survival rate under the paper protocol (as in the MVTracker evaluator)
DATASET_SURVIVAL = {
    'kubric-multiview':  0.5,
    'dex_ycb':           0.1,
    'cmupanoptic_3dgs':  1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output-root',  required=True)
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--split', default='test')
    parser.add_argument('--depth-subdir', type=str, default=None,
                        help="Depth subdir tag used during inference (e.g. 'duster-views-012345678' or 'depths'). "
                             "Must match what was passed to mvtracker_inference.py --depth-subdir.")
    parser.add_argument('--thresholds', nargs='+', type=float, default=None)
    parser.add_argument('--legacy-metrics', action='store_true',
                        help='score with get_eval_metrics (pooled over all points, all frames) instead of the '
                             'default paper-protocol metrics (get_eval_metrics_paper)')
    parser.add_argument('--survival-threshold', type=float, default=None,
                        help='L2 failure threshold for survival rate (default: the dataset value of the paper '
                             'protocol, 50 with --legacy-metrics)')
    parser.add_argument('--prefix', type=str, default='mvtracker_')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--csv-out',     type=str, default=None)
    parser.add_argument('--summary-out', type=str, default=None)
    return parser.parse_args()


def find_trials(output_root, datasets=None):
    root = Path(output_root)
    for pred_path in sorted(root.glob('*/*.npz')):
        dataset_name, filename = pred_path.relative_to(root).parts
        if datasets and dataset_name not in datasets:
            continue
        tid = filename[:-4]   # strip .npz
        yield dataset_name, tid, pred_path


def _cam_indices_from_metadata(metadata_path, cam_names_used):
    """Return the integer indices of cam_names_used within the sorted full camera list."""
    with open(metadata_path) as f:
        meta = yaml.safe_load(f)
    all_names = list(meta['intrinsic_matrices'].keys())
    if all(n.isdigit() for n in all_names):
        all_names = sorted(all_names, key=int)
    else:
        all_names = sorted(all_names)
    return [all_names.index(n) for n in cam_names_used]


def load_ground_truth(pose_path, frame_numbers, valid_mask, cam_indices=None):
    data = np.load(pose_path)
    coords_gt  = data['pose']
    vis_gt_raw = data['vis'] if 'vis' in data else None

    S, T_full, K, _ = coords_gt.shape
    frame_nums_clipped = np.clip(frame_numbers, 0, T_full - 1)

    coords_flat = coords_gt.transpose(0, 2, 1, 3).reshape(S * K, T_full, 3)
    coords_flat = coords_flat[valid_mask]
    coords_true = coords_flat[:, frame_nums_clipped, :].transpose(1, 0, 2)[np.newaxis]

    if vis_gt_raw is not None:
        vis_sub  = vis_gt_raw[..., cam_indices] if cam_indices is not None else vis_gt_raw
        vis_agg  = vis_sub.any(axis=-1)
        vis_flat = vis_agg.transpose(0, 2, 1).reshape(S * K, T_full)
        vis_flat = vis_flat[valid_mask]
        vis_true = vis_flat[:, frame_nums_clipped].T[np.newaxis, :, :, np.newaxis]
    else:
        vis_true = np.all(np.isfinite(coords_true), axis=-1, keepdims=True)

    return coords_true.astype(np.float32), vis_true.astype(bool)


def run_trial(pred_path, pose_path, thresholds, survival_threshold, prefix, dataset_name=None, legacy=False):
    pred  = np.load(pred_path)
    keys  = set(pred.files)

    # Support both posetail format (coords_pred/vis_pred embedded GT) and
    # legacy format (coords_3d/visibility with GT loaded from pose_path).
    if 'coords_pred' in keys:
        # Posetail format — GT already embedded
        coords_pred = torch.from_numpy(np.asarray(pred['coords_pred']).astype(np.float32))
        vis_pred    = torch.from_numpy(np.asarray(pred['vis_pred']).astype(np.float32))
        coords_true = torch.from_numpy(np.asarray(pred['coords_true']).astype(np.float32))
        if 'vis_true_cams' in keys:
            vtc = np.asarray(pred['vis_true_cams']).astype(np.float64)
            known_visible = np.isfinite(vtc) & (vtc > 0.5)
            vis_true = torch.from_numpy(known_visible.any(axis=-1, keepdims=True))
        else:
            vis_true = torch.from_numpy(np.asarray(pred['vis_true']).astype(bool))
        query_times = torch.from_numpy(np.asarray(pred['query_times'])) if 'query_times' in keys else None
    else:
        # Legacy format — load GT from pose_path
        coords_3d     = pred['coords_3d']      # (T, N, 3)
        visibility    = pred['visibility']     # (T, N) bool
        frame_numbers = pred['frame_numbers']
        valid_mask    = pred['valid_mask']

        cam_indices = None
        if 'cam_names' in keys:
            cam_names_used = pred['cam_names'].tolist()
            metadata_path  = Path(pose_path).parent / 'metadata.yaml'
            if metadata_path.exists():
                cam_indices = _cam_indices_from_metadata(str(metadata_path), cam_names_used)

        finite_mask = np.isfinite(coords_3d).all(axis=-1)
        visibility  = visibility & finite_mask
        coords_3d   = np.where(finite_mask[:, :, np.newaxis], coords_3d, 0.0)

        coords_pred = torch.from_numpy(coords_3d[np.newaxis].astype(np.float32))
        vis_logits  = np.where(visibility, 10.0, -10.0).astype(np.float32)
        vis_pred    = torch.from_numpy(vis_logits[np.newaxis, :, :, np.newaxis])

        coords_true_np, vis_true_np = load_ground_truth(
            pose_path, frame_numbers, valid_mask, cam_indices=cam_indices)
        coords_true = torch.from_numpy(coords_true_np)
        vis_true    = torch.from_numpy(vis_true_np)
        query_times = None

    if not legacy:
        # paper protocol (tracktail/model/eval_metrics_paper.py): per track, query frame and earlier frames not
        # evaluated, static/dynamic mean on kubric / dex_ycb. outputs without query_times were queried at frame 0
        return get_eval_metrics_paper(
            coords_pred=coords_pred.numpy(),
            coords_true=coords_true.numpy(),
            vis_pred=vis_pred.numpy(),
            vis_true=vis_true.numpy(),
            query_times=query_times.numpy() if query_times is not None else None,
            thresholds=thresholds,
            survival_threshold=survival_threshold,
            movement_thresholds=PAPER_MOVEMENT_THRESHOLDS.get(dataset_name),
            prefix=prefix,
        )

    return get_eval_metrics(
        vis_pred=vis_pred,
        vis_true=vis_true,
        coords_pred=coords_pred,
        coords_true=coords_true,
        thresholds=thresholds,
        survival_threshold=survival_threshold,
        prefix=prefix,
        query_times=query_times,
    )


def main():
    args = parse_args()

    tag    = f"_{args.depth_subdir}" if args.depth_subdir else ""
    trials = list(find_trials(args.output_root, datasets=args.datasets))
    if not trials:
        print(f'No prediction .npz files found under {args.output_root}')
        return

    print(f'Found {len(trials)} trial(s)')
    rows_by_dataset = {}
    n_computed = n_skipped = n_failed = 0

    for dataset_name, tid, pred_path in tqdm(trials, desc='Trials'):
        metrics_out = pred_path.with_name(pred_path.stem + '_metrics.npz')
        thresholds  = args.thresholds or DATASET_THRESHOLDS.get(dataset_name)

        if thresholds is None and dataset_name not in DATASET_THRESHOLDS:
            tqdm.write(
                f'  WARNING: no default thresholds for {dataset_name!r} — '
                f'using get_eval_metrics defaults.  Add to DATASET_THRESHOLDS or '
                f'pass --thresholds explicitly.'
            )

        # a metrics file of the other metric definition is recomputed (the paper metrics have an 'ate' entry)
        reuse = False
        if metrics_out.exists() and not args.force:
            existing = np.load(metrics_out, allow_pickle=True)
            reuse = (f'{args.prefix}ate' in existing.files) != args.legacy_metrics
        if reuse:
            metrics  = {k: float(existing[k]) for k in existing.files}
            n_skipped += 1
        else:
            # GT is embedded in the npz (new format); pose_path only needed for legacy fallback
            pred_keys = set(np.load(pred_path, allow_pickle=True).files)
            pose_path = None
            if 'coords_pred' not in pred_keys:
                # Legacy format: reconstruct pose_path from tid (session_trial)
                session, trial = tid.split('_', 1)
                pose_path = (
                    Path(args.dataset_root)
                    / dataset_name / args.split / session / trial / 'pose3d.npz'
                )
                if not pose_path.exists():
                    tqdm.write(f'  MISSING GT: {dataset_name}/{tid}')
                    n_failed += 1
                    continue
            try:
                metrics = run_trial(
                    pred_path=pred_path,
                    pose_path=pose_path,
                    thresholds=thresholds,
                    survival_threshold=(args.survival_threshold if args.survival_threshold is not None
                                        else 50 if args.legacy_metrics else DATASET_SURVIVAL.get(dataset_name, 50)),
                    prefix=args.prefix,
                    dataset_name=dataset_name,
                    legacy=args.legacy_metrics,
                )
                np.savez(metrics_out, **{k: np.array(v) for k, v in metrics.items()})
                n_computed += 1
            except Exception as e:
                tqdm.write(f'  FAILED {dataset_name}/{tid}: {e}')
                traceback.print_exc()
                n_failed += 1
                continue

        row = {'dataset': dataset_name, 'trial': tid}
        row.update({k: float(v) for k, v in metrics.items()})
        rows_by_dataset.setdefault(dataset_name, []).append(row)

    print(f'\nDone: {n_computed} computed, {n_skipped} skipped, {n_failed} failed')

    if not rows_by_dataset:
        print('No results to aggregate.')
        return

    all_rows = []
    for dataset_name, rows in rows_by_dataset.items():
        df = pd.DataFrame(rows)
        id_cols     = ['dataset', 'trial']
        metric_cols = [c for c in df.columns if c not in id_cols]
        df = df[id_cols + metric_cols]
        all_rows.append(df)

        dataset_dir = Path(args.output_root) / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        csv_out     = args.csv_out     or str(dataset_dir / f'all_metrics{tag}.csv')
        summary_out = args.summary_out or str(dataset_dir / f'summary_metrics{tag}.csv')

        df.to_csv(csv_out, index=False)
        print(f'Saved per-trial CSV to {csv_out}')

        summary_df = df.groupby('dataset')[metric_cols].mean().reset_index()
        summary_df.to_csv(summary_out, index=False)
        print(f'Saved summary CSV   to {summary_out}')

    print('\n--- Mean per dataset ---')
    combined    = pd.concat(all_rows, ignore_index=True)
    metric_cols = [c for c in combined.columns if c not in ['dataset', 'trial']]
    summary     = combined.groupby('dataset')[metric_cols].mean()
    max_len     = max(len(k) for k in metric_cols)
    for dataset_name, row in summary.iterrows():
        print(f'\n  [{dataset_name}]')
        for k in metric_cols:
            print(f'    {k:<{max_len}}  {row[k]:.4f}')


if __name__ == '__main__':
    main()
