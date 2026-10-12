#!/usr/bin/env python
"""Full test-set evaluation with COMPLETE error metrics, in one script.

For each requested dataset, runs query-first inference (mvtracker/training convention -- the
default in run_inference) on every test trial and computes the full metric set via
get_eval_metrics (MTE, MPJPE, delta-avg + per-threshold, survival, occlusion-acc,
occlusion-acc-percam, AJ + per-threshold). Metrics use the corrected masking: non-finite GT
(cleaned (0,0,-1)->NaN placeholders) and pre-query frames are excluded via the per-point
query_times -- this now includes both occlusion metrics (frame counts, not the visibility gate,
so occluded frames still count). occlusion-acc-percam scores the model's per-camera visibility
logits (vis_pred_2d) against per-camera GT; it is NaN for predictions cached before this column
existed (regenerate with --force to populate it).

Crop setup (best for moving subjects): query-first runs through the windowed tracker, which
RE-CROPS each chunk to follow the subject (n_overlap sets the cadence) and expands the crop by
a causal motion margin (--no-motion-margin to disable). This replaced an earlier static-crop
path that held one crop over the whole clip -- so predictions cached by that old path are
stale; pass --force (or delete the npz cache) to regenerate them through the fixed crop.

Predictions already present in the output folder are REUSED (metrics recomputed from the saved
npz) instead of re-running inference -- so the model / wandb folder / GPU are only needed for
trials that must still be computed. Writes per-dataset metrics.json plus a combined summary.json
and metrics.md (mean + median tables).

Usage:
    # regenerate tables from existing predictions (no model/GPU needed):
    pixi run python scripts/eval_testset_metrics.py \\
        --datasets dex_ycb kubric-multiview cmupanoptic_3dgs --out <preds_dir>

    # compute missing predictions from a wandb run (latest checkpoint by default):
    pixi run python scripts/eval_testset_metrics.py --datasets dex_ycb \\
        --out <preds_dir> --wandb-folder <wandb_run_dir> [--checkpoint STEP] [--force]
"""
import argparse
import glob
import json
import os
import sys
import time

# reduce CUDA fragmentation on dense point sets (cmupanoptic) before torch is imported
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from posetail.inference.inference_utils import (                         # noqa: E402
    load_model_from_base_folder, run_inference, load_camera_group_from_metadata)
from posetail.posetail.eval_metrics import (                             # noqa: E402
    get_eval_metrics,
    compute_cam_depths,
    get_delta_x_avg_depth_relative,
    get_mte_cam,
    get_survival_rate_depth_relative,
    get_average_jaccard_depth_relative,
    get_delta_x_avg_world_coord,
    get_mte_world_coord,
    get_survival_rate_world_coord,
    get_average_jaccard_world_coord,
    get_l1_world_coord,
    get_delta_x_avg_pertraj,
    get_occlusion_accuracy_pertraj,
)

DEFAULT_WANDB = '/groups/karashchuk/home/karashchukl/results/posetail-finetuning-v3/wandb/run-20260628_134003-39yczenk'
DEFAULT_ROOT = '/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6'

# TAP-Vid default thresholds (δavg over {1,2,4,8,16}) used for any dataset not listed below.
DEFAULT_SETTINGS = dict(thresholds=[1, 2, 4, 8, 16], survival=50, n_views=None, max_kpts=2500)

# per-dataset overrides. thresholds/survival are dataset-specific world-unit scales;
# n_views/max_kpts cap inference (cmupanoptic_3dgs is dense + many-camera).
SETTINGS = {
    'dex_ycb':          dict(thresholds=[0.01, 0.02, 0.05, 0.10, 0.20], survival=0.10,
                             n_views=None, max_kpts=2500),
    'kubric-multiview': dict(thresholds=[0.05, 0.1, 0.2, 0.4, 0.8],     survival=0.50,
                             n_views=None, max_kpts=2500),
    'cmupanoptic_3dgs': dict(thresholds=[0.05, 0.10, 0.20, 0.40],       survival=1.0,
                             n_views=4, max_kpts=600),
    # TAPVid-3D fixed-metric thresholds: 1cm, 4cm, 16cm, 64cm, 256cm
    # (supplemental paper values — 4× progression matching the depth-adaptive pixel thresholds).
    # Depth-relative cam/world metrics are added separately via eval_outputs_cam.
    'point-odyssey-human':     dict(thresholds=[0.30, 0.60, 1.20, 2.40, 4.80], survival=50,
                                    n_views=None, max_kpts=2500),
    'po-animal-sub256-motion': dict(thresholds=[0.30, 0.60, 1.20, 2.40, 4.80], survival=50,
                                    n_views=None, max_kpts=2500),
    'tapvid3d_adt':       dict(thresholds=[0.01, 0.04, 0.16, 0.64, 2.56], survival=0.50,
                               n_views=None, max_kpts=2500),
    'tapvid3d_drivetrack': dict(thresholds=[0.01, 0.04, 0.16, 0.64, 2.56], survival=0.50,
                                n_views=None, max_kpts=2500),
    'tapvid3d_pstudio':   dict(thresholds=[0.01, 0.04, 0.16, 0.64, 2.56], survival=0.50,
                               n_views=None, max_kpts=2500),
}
METRIC_KEYS = ['mte', 'mpjpe', 'delta_x_avg', 'survival_rate', 'occlusion_acc',
               'occlusion_acc_percam', 'avg_jaccard', 'l1']

# TAPVid-3D depth-relative metric config (cam-coord and world-coord).
# Thresholds from the TAPVid-3D supplemental; a point passes when dist/Z_depth < threshold.
# cam-coord: Z_depth = per-point per-frame Z_cam.
# world-coord: Z_depth = per-trajectory mean Z_cam (D4RT addition; pstudio excluded).
TAPVID3D_DATASETS      = {'tapvid3d_adt', 'tapvid3d_drivetrack', 'tapvid3d_pstudio'}
TAPVID3D_WORLD_DATASETS = {'tapvid3d_adt', 'tapvid3d_drivetrack'}   # static cam excluded
TAPVID3D_THRESHOLDS    = [0.01, 0.04, 0.16, 0.64, 2.56]
TAPVID3D_SURVIVAL_DEPTH = 0.50                                        # 50% of depth
METRIC_KEYS_CAM   = ['mte_cam',   'delta_x_avg_cam',   'survival_rate_cam',   'avg_jaccard_cam',   'occlusion_acc']
METRIC_KEYS_WORLD = ['mte_world', 'delta_x_avg_world', 'survival_rate_world', 'avg_jaccard_world', 'l1_world']

# TAPVid-2D per-trajectory metrics (matching MVTracker "Our Metrics" convention).
# avg_jaccard is already per-trajectory in get_average_jaccard; listed here so the
# pertraj table is complete and directly comparable to Table E.1 in MVTracker.
TAPVID2D_DATASETS   = {'tapvid2d'}
METRIC_KEYS_PERTRAJ = ['delta_x_avg_pertraj', 'occlusion_acc_pertraj', 'avg_jaccard_pertraj']


def find_test_trials(dataset, root):
    # glob follows the v3->v2 symlink for cmupanoptic_3dgs.
    # Supports old convention (test/<subject>/trial/pose3d.npz) and tapvid layouts
    # (test/<seq>/<id>/pose3d.npz for tapvid3d, test/<cat>/<name>/pose2d.npz for tapvid2d).
    base = os.path.join(root, dataset, 'test')
    pose3d = glob.glob(os.path.join(base, '**', 'pose3d.npz'), recursive=True)
    pose2d = glob.glob(os.path.join(base, '**', 'pose2d.npz'), recursive=True)
    return sorted(set(os.path.dirname(p) for p in pose3d + pose2d))


def _to_np(x):
    """Convert a torch tensor or array-like to a float64 numpy array."""
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def load_trial_camera_ext(trial_path):
    """Return ext for the first camera of a tapvid3d trial.

    ext: (4,4) static world→cam extrinsic, or (T,4,4) for moving-cam sequences.
         cam['ext'] is always a torch tensor after format_camera; converted to numpy.
    """
    metadata_path = os.path.join(trial_path, 'metadata.yaml')
    cgroup = load_camera_group_from_metadata(metadata_path)
    cam = cgroup[0]
    return _to_np(cam['ext'])      # (4,4) static or (T,4,4) moving


def eval_outputs_cam(out, trial_path, ds=None):
    """Compute depth-relative cam-coord (and optionally world-coord) metrics for a tapvid3d trial.

    Returns a dict of _cam / _world-suffixed keys to merge into the per-trial result.
    World-coord metrics are added only when ds is in TAPVID3D_WORLD_DATASETS.
    """
    keys = set(out.files) if hasattr(out, 'files') else set(out)
    cp = np.asarray(out['coords_pred'])          # (1, T, N, 3)
    ct = np.asarray(out['coords_true'])          # (1, T, N, 3)
    vp = np.asarray(out['vis_pred'])
    vt = np.asarray(out['vis_true']).astype(bool)
    qt = np.asarray(out['query_times']) if 'query_times' in keys else None

    ext = load_trial_camera_ext(trial_path)

    T = ct.shape[1]
    depths = compute_cam_depths(ct[0], ext)      # (T, N)
    depths_b = depths[np.newaxis]                # (1, T, N)

    valid = np.isfinite(ct).all(axis=-1, keepdims=True)
    if qt is not None:
        qt_arr = np.asarray(qt).reshape(1, -1)
        at_or_after = np.arange(T)[None, :, None] >= qt_arr[:, None, :]
        valid = valid & at_or_after[..., None]
    vis_eff = vt & valid                                   # (1, T, N, 1)

    dx_avg_c, dx_dict_c = get_delta_x_avg_depth_relative(
        cp, ct, vis_eff, TAPVID3D_THRESHOLDS, depths_b)
    m = {
        'mte_cam':           float(get_mte_cam(cp, ct, vis_eff, depths_b)),
        'delta_x_avg_cam':   float(dx_avg_c),
        'survival_rate_cam': float(get_survival_rate_depth_relative(
            cp, ct, vis_eff, depths_b, TAPVID3D_SURVIVAL_DEPTH)),
        'avg_jaccard_cam':   float(get_average_jaccard_depth_relative(
            cp, ct, vp, vis_eff, TAPVID3D_THRESHOLDS, depths_b)[0]),
    }
    for th, v in dx_dict_c.items():
        m[f'delta_x_{th}_cam'] = float(v)

    if ds in TAPVID3D_WORLD_DATASETS:
        dx_avg_w, dx_dict_w = get_delta_x_avg_world_coord(
            cp, ct, vis_eff, TAPVID3D_THRESHOLDS, depths_b)
        m.update({
            'mte_world':           float(get_mte_world_coord(cp, ct, vis_eff, depths_b)),
            'delta_x_avg_world':   float(dx_avg_w),
            'survival_rate_world': float(get_survival_rate_world_coord(
                cp, ct, vis_eff, depths_b, TAPVID3D_SURVIVAL_DEPTH)),
            'avg_jaccard_world':   float(get_average_jaccard_world_coord(
                cp, ct, vp, vis_eff, TAPVID3D_THRESHOLDS, depths_b)[0]),
            'l1_world':            float(get_l1_world_coord(cp, ct, vis_eff, depths_b)),
        })
        for th, v in dx_dict_w.items():
            m[f'delta_x_{th}_world'] = float(v)

    return m


def eval_outputs_pertraj(out, thresholds, avg_jaccard_val):
    """Per-trajectory variants of delta_x_avg, occlusion_acc, avg_jaccard for tapvid2d.

    Matches MVTracker's 'Our Metrics' convention (each track equally weighted).
    avg_jaccard_pertraj == avg_jaccard since get_average_jaccard is already per-trajectory;
    the caller passes avg_jaccard_val from the already-computed eval_outputs result.
    """
    keys = set(out.files) if hasattr(out, 'files') else set(out)
    cp = np.asarray(out['coords_pred'])
    ct = np.asarray(out['coords_true'])
    vp = np.asarray(out['vis_pred'])
    vt = np.asarray(out['vis_true']).astype(bool)
    qt = np.asarray(out['query_times']) if 'query_times' in keys else None

    T = ct.shape[1]
    valid = np.isfinite(ct).all(axis=-1, keepdims=True)
    if qt is not None:
        qt_arr = np.asarray(qt).reshape(1, -1)
        at_or_after = np.arange(T)[None, :, None] >= qt_arr[:, None, :]
        valid = valid & at_or_after[..., None]
    vis_eff = vt & valid

    dx_avg, _ = get_delta_x_avg_pertraj(cp, ct, vis_eff, thresholds)
    return {
        'delta_x_avg_pertraj':   float(dx_avg),
        'occlusion_acc_pertraj': float(get_occlusion_accuracy_pertraj(vp, vt, mask=valid)),
        'avg_jaccard_pertraj':   float(avg_jaccard_val),
    }


def _vis_true_from_cams(vis_true_cams):
    """OR-fuse per-camera GT visibility (NaN = unknown, excluded) across the camera axis
    (last dim) to get the global visible/occluded label for the cameras ACTUALLY USED at
    inference. An all-unknown frame has no evidence of visibility and is treated as occluded."""
    vtc = np.asarray(vis_true_cams).astype(np.float64)
    known_visible = np.isfinite(vtc) & (vtc > 0.5)
    return known_visible.any(axis=-1, keepdims=True)


def eval_outputs(out, thresholds, survival):
    """Compute the full metric set from a predictions dict (an in-memory outputs dict or an
    np.load NpzFile -- both index by key and expose .files/keys)."""
    keys = set(out.files) if hasattr(out, 'files') else set(out)
    cp = torch.as_tensor(np.asarray(out['coords_pred']), dtype=torch.float32)
    ct = torch.as_tensor(np.asarray(out['coords_true']), dtype=torch.float32)
    vp = torch.as_tensor(np.asarray(out['vis_pred']), dtype=torch.float32)
    qt = torch.as_tensor(np.asarray(out['query_times']), dtype=torch.long) \
        if 'query_times' in keys else None
    # Per-camera occlusion: model logits vs per-camera GT (both saved by run_inference when the
    # trial has visibility GT). Absent from predictions cached before this was added -> the
    # per-cam metric is left as NaN (regenerate with --force to populate it).
    vp2d = np.asarray(out['vis_pred_2d']) if 'vis_pred_2d' in keys else None
    vtc = np.asarray(out['vis_true_cams']) if 'vis_true_cams' in keys else None
    # Global vis_true is recomputed on the fly from vis_true_cams (OR over the cameras actually
    # used at inference) rather than trusting the cached 'vis_true' array: predictions cached
    # before the inference_utils.py fix baked in a vis_true OR'd over the FULL camera rig,
    # independent of --n-views subsampling -- which desyncs the global occlusion_acc from
    # occlusion_acc_percam and from what the model actually saw. Recomputing here fixes stale
    # cached predictions retroactively, with no need to rerun inference; it is a no-op for
    # fresh predictions where the saved vis_true is already consistent.
    if vtc is not None:
        vt = torch.as_tensor(_vis_true_from_cams(vtc), dtype=torch.bool)
    else:
        vt = torch.as_tensor(np.asarray(out['vis_true']), dtype=torch.bool)
    m = get_eval_metrics(vp, vt, cp, ct, thresholds=thresholds,
                         survival_threshold=survival, prefix='', query_times=qt,
                         vis_pred_2d=vp2d, vis_true_cams=vtc)
    m = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in m.items()}
    m.setdefault('occlusion_acc_percam', float('nan'))
    m['n_kpts'] = int(cp.shape[2])
    m['n_late_query'] = int((np.asarray(out['query_times']) > 0).sum()) if 'query_times' in keys else 0
    return m


def fmt_dur(seconds):
    s = int(round(seconds)); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f'{h}:{m:02d}:{sec:02d}' if h else f'{m}:{sec:02d}'


def write_report(all_summary, out_dir):
    """Write combined summary.json + metrics.md (mean and median tables) over all datasets."""
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(all_summary, f, indent=2)

    def table(agg):
        # include l1_world as an optional extra column when present for any dataset
        extra = ['l1_world'] if any(f'l1_world_{agg}' in s for s in all_summary.values()) else []
        all_keys = METRIC_KEYS + extra
        L = ['| dataset | #tr | ' + ' | '.join(k.replace('_', '-') for k in all_keys) + ' |',
             '|---|---|' + '---|' * len(all_keys)]
        for ds, s in all_summary.items():
            vals = ' | '.join(
                f'{s[f"{k}_{agg}"]:.4f}' if f'{k}_{agg}' in s else 'nan'
                for k in all_keys)
            L.append(f'| {ds} | {s["n_trials"]} | {vals} |')
        return '\n'.join(L)

    tapvid_ds = [ds for ds in all_summary if ds in TAPVID3D_DATASETS
                 and f'{METRIC_KEYS_CAM[0]}_mean' in all_summary[ds]]
    tapvid_world_ds = [ds for ds in tapvid_ds if ds in TAPVID3D_WORLD_DATASETS
                       and f'{METRIC_KEYS_WORLD[0]}_mean' in all_summary[ds]]
    tapvid2d_ds = [ds for ds in all_summary if ds in TAPVID2D_DATASETS
                   and f'{METRIC_KEYS_PERTRAJ[0]}_mean' in all_summary[ds]]

    def table_depth(agg, keys, ds_list):
        if not ds_list:
            return '_none_'
        L = ['| dataset | #tr | ' + ' | '.join(k.replace('_', '-') for k in keys) + ' |',
             '|---|---|' + '---|' * len(keys)]
        for ds in ds_list:
            s = all_summary[ds]
            L.append(f'| {ds} | {s["n_trials"]} | '
                     + ' | '.join(f'{s[f"{k}_{agg}"]:.4f}' for k in keys) + ' |')
        return '\n'.join(L)

    depth_note = (f'Thresholds {TAPVID3D_THRESHOLDS}; a point passes when '
                  f'dist/Z_depth < threshold. Survival threshold {TAPVID3D_SURVIVAL_DEPTH}.')

    cfg_rows = ['| dataset | n_views | view_seed |', '|---|---|---|']
    for ds, s in all_summary.items():
        n_views = s.get('n_views')
        cfg_rows.append(f'| {ds} | {n_views if n_views is not None else "all"} | {s.get("view_seed", 0)} |')

    md = ['# Test-set error metrics (query-first)', '',
          'Per-point first-visible query anchoring; pre-query + non-finite GT masked. Metrics via '
          '`get_eval_metrics`, aggregated over test trials.', '',
          '## Camera sampling config', '', '\n'.join(cfg_rows), '',
          '## Mean over trials', '', table('mean'), '',
          '## Median over trials', '', table('median'), '',
          'Thresholds (world units): '
          + '; '.join(f'{ds} {SETTINGS.get(ds, DEFAULT_SETTINGS)["thresholds"]} '
                      f'surv={SETTINGS.get(ds, DEFAULT_SETTINGS)["survival"]}'
                      for ds in all_summary)
          + '. Per-trial breakdowns in `<dataset>/metrics.json`.',
          '',
          '## TAPVid-3D camera-coordinate metrics (per-frame depth, mean)', '',
          depth_note + ' Z_depth = per-point per-frame Z_cam.', '',
          table_depth('mean', METRIC_KEYS_CAM, tapvid_ds), '',
          '## TAPVid-3D camera-coordinate metrics (per-frame depth, median)', '',
          table_depth('median', METRIC_KEYS_CAM, tapvid_ds), '',
          '## TAPVid-3D world-coordinate metrics (per-trajectory depth, mean)', '',
          depth_note + ' Z_depth = per-trajectory mean Z_cam (adt + drivetrack only).', '',
          table_depth('mean', METRIC_KEYS_WORLD, tapvid_world_ds), '',
          '## TAPVid-3D world-coordinate metrics (per-trajectory depth, median)', '',
          table_depth('median', METRIC_KEYS_WORLD, tapvid_world_ds), '',
          '## TAPVid-2D per-trajectory metrics (MVTracker "Our Metrics", mean)', '',
          'Each track weighted equally (vs all-points-pooled in the standard table above). '
          'avg-jaccard-pertraj == avg-jaccard (get_average_jaccard is already per-trajectory).', '',
          table_depth('mean', METRIC_KEYS_PERTRAJ, tapvid2d_ds), '',
          '## TAPVid-2D per-trajectory metrics (MVTracker "Our Metrics", median)', '',
          table_depth('median', METRIC_KEYS_PERTRAJ, tapvid2d_ds)]
    with open(os.path.join(out_dir, 'metrics.md'), 'w') as f:
        f.write('\n'.join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True,
                    help='which datasets to evaluate; unknown datasets use TAP-Vid defaults '
                         f'(thresholds={DEFAULT_SETTINGS["thresholds"]}, '
                         f'survival={DEFAULT_SETTINGS["survival"]})')
    ap.add_argument('--out', required=True,
                    help='predictions/output folder: <out>/<dataset>/<trial>.npz (reused if present)')
    ap.add_argument('--root', default=DEFAULT_ROOT,
                    help='dataset root: expects <root>/<dataset>/test/.../trial/pose3d.npz')
    ap.add_argument('--wandb-folder', default=DEFAULT_WANDB,
                    help='wandb run dir (files/config.toml + files/checkpoints/); only loaded '
                         'when a trial needs inference')
    ap.add_argument('--checkpoint', type=int, default=None,
                    help='checkpoint step; default = latest checkpoint in the wandb folder')
    ap.add_argument('--force', action='store_true', help='recompute even if predictions exist')
    ap.add_argument('--n-frames', type=int, default=500, help='cap; uses all available frames')
    ap.add_argument('--max-kpts', type=int, default=None, help='override per-dataset max_kpts')
    ap.add_argument('--n-views', type=int, default=None, help='override per-dataset n_views')
    ap.add_argument('--view-seed', type=int, default=0)
    ap.add_argument('--cam-names', nargs='+', default=None,
                    help='Explicit camera names to use; overrides --n-views')
    ap.add_argument('--pose-file', type=str, default=None,
                    help='Override the pose file name (e.g. pose3d_100.npz; default: pose3d.npz)')
    # Crop setup: query-first runs through the windowed tracker, which re-crops every chunk
    # to FOLLOW the subject (the fix for the static-crop regression). motion-margin expands
    # each chunk's crop by the previous chunk's velocity so fast subjects (dex_ycb) stay
    # in-frame; it is ~inert for slow subjects. Both default ON = best crop setup.
    ap.add_argument('--n-overlap', type=int, default=8,
                    help='re-crop / re-anchor cadence: window advances by (clip_len - n_overlap) '
                         'frames, so a smaller value re-crops more often for very fast subjects')
    ap.add_argument('--no-motion-margin', dest='motion_margin', action='store_false', default=True,
                    help='disable the causal motion-margin crop expansion (default ON)')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Lazy model load: only pay for the wandb folder / GPU when a trial must be computed.
    model_state = {}

    def get_model():
        if not model_state:
            device = torch.device(args.device)
            print(f'Loading model from {args.wandb_folder} '
                  f'(checkpoint={"latest" if args.checkpoint is None else args.checkpoint}) ...')
            model, _cfg, cfg_path, ckpt = load_model_from_base_folder(
                args.wandb_folder, checkpoint=args.checkpoint, device=device)
            print(f'Model loaded: {ckpt}')
            model_state.update(model=model, config_path=cfg_path, checkpoint_path=ckpt, device=device)
        return model_state

    # Build the full work list up front so we can show a global counter + an ETA that knows
    # which trials are cached (fast) vs need inference (slow).
    work = []
    for ds in args.datasets:
        cfg = SETTINGS.get(ds, DEFAULT_SETTINGS)
        n_views = args.n_views if args.n_views is not None else cfg['n_views']
        max_kpts = args.max_kpts if args.max_kpts is not None else cfg['max_kpts']
        ds_out = os.path.join(args.out, ds); os.makedirs(ds_out, exist_ok=True)
        test_base = os.path.join(args.root, ds, 'test')
        for tp in find_test_trials(ds, args.root):
            tid = os.path.relpath(tp, test_base).replace(os.sep, '_')
            npz = os.path.join(ds_out, f'{tid}.npz')
            work.append(dict(ds=ds, cfg=cfg, n_views=n_views, max_kpts=max_kpts, tp=tp,
                             tid=tid, npz=npz, cam_names=args.cam_names,
                             cached=os.path.exists(npz) and not args.force))
    total = len(work)
    n_cached = sum(w['cached'] for w in work)
    print(f'\n{total} trials across {len(args.datasets)} dataset(s): '
          f'{n_cached} cached, {total - n_cached} to infer')
    for ds in args.datasets:
        ws = [w for w in work if w['ds'] == ds]
        cam_label = ('cam_names=' + str(ws[0]['cam_names'])) if (ws and ws[0]['cam_names']) \
            else f'n_views={ws[0]["n_views"] if ws else "-"}'
        print(f'  {ds}: {len(ws)} trials ({cam_label}, '
              f'max_kpts={ws[0]["max_kpts"] if ws else "-"}, '
              f'{sum(w["cached"] for w in ws)} cached)')

    # Running-mean per-trial durations (seeded with rough priors) -> ETA over remaining work,
    # counting each remaining trial as cached (~instant) or infer (slow) since we know which.
    avg = {'cached': 1.0, 'infer': 90.0}
    seen = {'cached': 0, 'infer': 0}
    results = {ds: [] for ds in args.datasets}
    t_start = time.time()
    for i, w in enumerate(work):
        ds, cfg, tid = w['ds'], w['cfg'], w['tid']
        t0 = time.time()
        try:
            if w['cached']:
                with np.load(w['npz'], allow_pickle=True) as data:
                    m = eval_outputs(data, cfg['thresholds'], cfg['survival'])
                    if ds in TAPVID3D_DATASETS:
                        m.update(eval_outputs_cam(data, w['tp'], ds))
                    if ds in TAPVID2D_DATASETS:
                        m.update(eval_outputs_pertraj(data, cfg['thresholds'], m['avg_jaccard']))
                    if 'cam_indices_used' in data:
                        m['cam_indices_used'] = data['cam_indices_used'].tolist()
                    if 'cam_names_used' in data:
                        m['cam_names_used'] = data['cam_names_used'].tolist()
                src = 'cached'
            else:
                ms = get_model()
                out = run_inference(
                    model=ms['model'], config_path=ms['config_path'],
                    checkpoint_path=ms['checkpoint_path'], trial_path=w['tp'], start_frame=0,
                    n_frames=args.n_frames, n_overlap=args.n_overlap, per_subject=True,
                    device=ms['device'], max_kpts=w['max_kpts'], n_views=w['n_views'],
                    seed=args.view_seed, outpath=w['npz'], query_first=True,
                    motion_margin=args.motion_margin, cam_names_filter=w['cam_names'],
                    pose_file=args.pose_file)
                torch.cuda.empty_cache()
                m = eval_outputs(out, cfg['thresholds'], cfg['survival'])
                if ds in TAPVID3D_DATASETS:
                    m.update(eval_outputs_cam(out, w['tp'], ds))
                if ds in TAPVID2D_DATASETS:
                    m.update(eval_outputs_pertraj(out, cfg['thresholds'], m['avg_jaccard']))
                if 'cam_indices_used' in out:
                    m['cam_indices_used'] = np.asarray(out['cam_indices_used']).tolist()
                if 'cam_names_used' in out:
                    m['cam_names_used'] = np.asarray(out['cam_names_used']).tolist()
                src = 'infer'
            m['trial'] = tid; m['status'] = 'ok'
            results[ds].append(m)
            info = (f'mte={m["mte"]:.4f} dx={m["delta_x_avg"]:.3f} surv={m["survival_rate"]:.3f} '
                    f'aj={m["avg_jaccard"]:.3f} K={m["n_kpts"]}')
        except Exception as exc:                    # noqa: BLE001
            results[ds].append({'trial': tid, 'status': 'FAILED', 'err': str(exc)})
            src, info = 'FAILED', str(exc).splitlines()[0][:80]

        dt = time.time() - t0
        kind = 'cached' if w['cached'] else 'infer'
        seen[kind] += 1
        avg[kind] += (dt - avg[kind]) / seen[kind]        # running mean
        eta = sum(avg['cached' if r['cached'] else 'infer'] for r in work[i + 1:])
        print(f'[{i + 1:>3}/{total}] {ds}/{tid} [{src}] {info} | {dt:4.0f}s  '
              f'elapsed {fmt_dur(time.time() - t_start)}  eta {fmt_dur(eta)}', flush=True)

    # per-dataset aggregation + save
    all_summary = {}
    for ds in args.datasets:
        per_trial = results[ds]
        ok = [r for r in per_trial if r['status'] == 'ok']
        ws = [w for w in work if w['ds'] == ds]
        ds_n_views = ws[0]['n_views'] if ws else None
        summary = {'n_trials': len(ok), 'n_views': ds_n_views, 'view_seed': args.view_seed}
        for k in METRIC_KEYS:
            vals = np.array([r[k] for r in ok], dtype=float)
            summary[f'{k}_mean'] = float(np.nanmean(vals)) if vals.size else float('nan')
            summary[f'{k}_median'] = float(np.nanmedian(vals)) if vals.size else float('nan')
        if ds in TAPVID3D_DATASETS:
            depth_keys = METRIC_KEYS_CAM + (METRIC_KEYS_WORLD if ds in TAPVID3D_WORLD_DATASETS else [])
            for k in depth_keys:
                vals = np.array([r.get(k, float('nan')) for r in ok], dtype=float)
                summary[f'{k}_mean'] = float(np.nanmean(vals)) if vals.size else float('nan')
                summary[f'{k}_median'] = float(np.nanmedian(vals)) if vals.size else float('nan')
        if ds in TAPVID2D_DATASETS:
            for k in METRIC_KEYS_PERTRAJ:
                vals = np.array([r.get(k, float('nan')) for r in ok], dtype=float)
                summary[f'{k}_mean'] = float(np.nanmean(vals)) if vals.size else float('nan')
                summary[f'{k}_median'] = float(np.nanmedian(vals)) if vals.size else float('nan')
        all_summary[ds] = summary
        with open(os.path.join(args.out, ds, 'metrics.json'), 'w') as f:
            json.dump({'summary': summary, 'per_trial': per_trial}, f, indent=2)
        print(f'--- {ds} over {len(ok)} trials: '
              + '  '.join(f'{k}={summary[f"{k}_mean"]:.4f}' for k in METRIC_KEYS))

    print(f'\nTotal wall time: {fmt_dur(time.time() - t_start)}')
    write_report(all_summary, args.out)
    print('\n========== TEST-SET SUMMARY (mean / median over trials) ==========')
    hdr = f'{"dataset":18s} ' + ' '.join(f'{k:>13s}' for k in METRIC_KEYS)
    print(hdr)
    for ds, s in all_summary.items():
        print(f'{ds:18s} ' + ' '.join(f'{s[f"{k}_mean"]:13.4f}' for k in METRIC_KEYS))
    print(f'\nSaved: {os.path.join(args.out, "summary.json")} + metrics.md')


if __name__ == '__main__':
    main()
