"""Compute per-frame survival rate curves and save to CSV.

For each (model, dataset) combination, loads all trial prediction files,
computes survival rate at every frame, and writes the mean ± std across
trials to survival_curves.csv.

Survival rate at frame t = fraction of tracked points whose cumulative L2
error has not yet exceeded `THRESHOLD` at any visible frame up to t.
Points are only counted from their query_time onward (pre-query frames are
excluded). The GT must be finite at a frame for it to count as a failure.

A track that CoTracker could not triangulate in a frame has no prediction there (nan). --nan-policy decides what that
means for the survival: "skip" (default) ignores the frame, as the distance metrics (mte, ate) do; "fail" counts a
frame that is visible in the ground truth but has no prediction as a failure.

Usage:
    python compute_survival.py [--out survival_curves.csv] [--cams 4 9] [--nan-policy skip|fail]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import yaml


THRESHOLD = 50.0  # world-unit L2 failure distance (matches eval_testset_metrics.py)

RESULTS = "/groups/karashchuk/home/ruppk2/results/paper-final-v2"
DATASET_ROOT = "/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6"

DATASETS = {   # name in the output, dataset folder, trial folder prefix
    "human": ("point-odyssey-human", "hum"),
    "animal": ("po-animal-sub256-motion", "ani"),
}


def make_configs(cams):
    """The runs of cotracker_cached_po.sh, mvtracker_cached_po.sh and posetail_cached_po.sh with n cameras."""
    configs = []
    for n in cams:
        for dataset, (folder, prefix) in DATASETS.items():
            configs.append(dict(
                model=f"posetail-{n}cams", dataset=dataset,
                pattern=f"{RESULTS}/posetail-outputs/main_finetune_vzkjqrys_po_{n}cams/{folder}/{prefix}*_trial.npz"))
            configs.append(dict(
                model=f"cotracker-{n}cams", dataset=dataset, cotracker=True,
                pattern=f"{RESULTS}/cotracker-outputs/paper-po-{n}cams/{folder}/test/{prefix}*/trial/cotracker_3d.npz"))
            for depth in ("gt", "duster"):
                configs.append(dict(
                    model=f"mvtracker-{n}cams-{depth}", dataset=dataset,
                    pattern=f"{RESULTS}/mvtracker-outputs/paper-po-{n}cams-{depth}/{folder}/{prefix}*_trial.npz"))
    return configs


def cam_indices(metadata_path, cam_names):
    """Positions of cam_names in the camera list of metadata.yaml (the columns of vis)."""
    with open(metadata_path) as f:
        names = list(yaml.safe_load(f)["intrinsic_matrices"].keys())
    names = sorted(names, key=int) if all(n.isdigit() for n in names) else sorted(names)
    return [names.index(n) for n in cam_names]


def load_trial(path, cotracker=False, pose_file="pose3d.npz"):
    """Load (coords_pred, coords_true, vis_true, query_times) from a trial file.

    All formats store coords_pred (1,T,N,3), coords_true (1,T,N,3), vis_true (1,T,N,1) and query_times (absent in
    cotracker files of runs without --query-first: all points queried at frame 0).
    vis_true is visibility in any camera of the pose file. For a run with fewer cameras it has to be restricted to
    the cameras that were used: posetail/mvtracker files store that as vis_true_cams (1,T,N,C_used); the cotracker
    files do not, so it is rebuilt from the pose file of the trial for the cameras in cam_names.

    Returns arrays of shape:
        coords_pred : (T, N, 3)
        coords_true : (T, N, 3)
        vis_true    : (T, N)  bool
        query_times : (N,)    int
    """
    d = np.load(path, allow_pickle=True)
    coords_pred = np.asarray(d["coords_pred"]).squeeze(0)               # (T, N, 3)
    coords_true = np.asarray(d["coords_true"]).squeeze(0)               # (T, N, 3)
    vis_true = np.asarray(d["vis_true"]).squeeze((0, -1)).astype(bool)  # (T, N)
    if "vis_true_cams" in d.files:
        vtc = np.asarray(d["vis_true_cams"]).squeeze(0).astype(np.float64)   # (T, N, C_used)
        vis_true = (np.isfinite(vtc) & (vtc > 0.5)).any(axis=-1)
    elif cotracker and "cam_names" in d.files:
        trial_dir = os.path.dirname(path)                                    # .../<dataset>/test/<session>/trial
        pose_dir = os.path.join(DATASET_ROOT, *trial_dir.split(os.sep)[-4:])
        vis_all = np.load(os.path.join(pose_dir, pose_file))["vis"][0]       # (T_full, N_full, C)
        idx = cam_indices(os.path.join(pose_dir, "metadata.yaml"), [str(c) for c in d["cam_names"]])
        vis_true = vis_all[..., idx].any(-1)[d["frame_numbers"]][:, d["valid_mask"]]
    if "query_times" in d.files:
        query_times = np.asarray(d["query_times"]).astype(np.int32)
    else:
        query_times = np.zeros(coords_pred.shape[1], dtype=np.int32)
    return coords_pred, coords_true, vis_true, query_times


def survival_rate_per_frame(coords_pred, coords_true, vis_true, query_times, threshold, nan_policy="skip"):
    """Fraction of tracks still alive at each frame.

    A track fails at the first visible frame (at or after its query_time) where
    the L2 distance to GT exceeds `threshold`. GT must also be finite to count
    as a failure. If a track never fails it survives to frame T (counted as alive
    at all frames). Tracks with no valid visible frames are excluded entirely.
    Frames without a prediction (nan, a failed triangulation) are ignored with nan_policy "skip" and count as a failure
    with "fail".

    Returns
    -------
    per_frame : (T,) float in [0, 1]
    """
    T, N = vis_true.shape
    dist = np.linalg.norm(coords_pred - coords_true, axis=-1)  # (T, N)
    gt_finite = np.isfinite(coords_true).all(axis=-1)           # (T, N)

    failure_frames = []
    for n in range(N):
        qt = int(query_times[n])
        # only frames at or after query_time, where GT is visible and finite
        active = vis_true[qt:, n] & gt_finite[qt:, n]
        if not np.any(active):
            continue
        too_far = dist[qt:, n] > threshold                                   # False where the prediction is nan
        if nan_policy == "fail":
            too_far = too_far | ~np.isfinite(dist[qt:, n])
        failed_rel = np.where(too_far & active)[0]
        failure_frames.append(int(failed_rel[0]) + qt if len(failed_rel) > 0 else T)

    if not failure_frames:
        return np.full(T, np.nan)

    fa = np.array(failure_frames)
    return np.array([np.mean(fa > t) for t in range(T)])


def process_config(cfg, threshold, nan_policy="skip"):
    paths = sorted(glob.glob(cfg["pattern"]))
    if not paths:
        print(f"  WARNING: no files found for {cfg['model']} / {cfg['dataset']}")
        return []

    curves = []
    for path in paths:
        try:
            cp, ct, vt, qt = load_trial(path, cotracker=cfg.get("cotracker", False))
            curve = survival_rate_per_frame(cp, ct, vt, qt, threshold, nan_policy)
            curves.append(curve)
        except Exception as e:
            print(f"  ERROR loading {path}: {e}")

    if not curves:
        return []

    # Align to shortest sequence (handles 256 vs 499 frame mismatch within a dataset)
    T = min(len(c) for c in curves)
    curves = np.stack([c[:T] for c in curves], axis=0)  # (n_trials, T)

    rows = []
    mean = np.nanmean(curves, axis=0)
    std = np.nanstd(curves, axis=0)
    for t in range(T):
        rows.append(dict(
            model=cfg["model"],
            dataset=cfg["dataset"],
            frame=t,
            survival_mean=float(mean[t]),
            survival_std=float(std[t]),
            n_trials=curves.shape[0],
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="survival_curves.csv")
    ap.add_argument("--cams", type=int, nargs="+", default=[4, 9],
                    help="Number of cameras of the runs to process (the paper-po-<n>cams folders)")
    ap.add_argument("--nan-policy", choices=["skip", "fail"], default="skip",
                    help="skip: ignore frames without a prediction; fail: they count as a failure when GT is visible")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Single threshold (overrides --thresholds)")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.30, 0.60, 1.20, 2.40, 4.80, 15.00],
                    help="One or more failure thresholds (world units)")
    args = ap.parse_args()

    thresholds = [args.threshold] if args.threshold is not None else args.thresholds

    configs = make_configs(args.cams)
    all_rows = []
    for thr in thresholds:
        print(f"\n=== Threshold {thr} ===")
        for cfg in configs:
            print(f"Processing {cfg['model']} / {cfg['dataset']} ...")
            rows = process_config(cfg, thr, args.nan_policy)
            for r in rows:
                r["threshold"] = thr
            all_rows.extend(rows)
            if rows:
                T = max(r["frame"] for r in rows) + 1
                final = rows[-1]
                print(f"  {final['n_trials']} trials, {T} frames, "
                      f"final survival={final['survival_mean']:.3f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
