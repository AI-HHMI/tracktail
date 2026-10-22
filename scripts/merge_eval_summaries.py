"""Rebuild a combined summary.json + metrics.md from the per-dataset metrics.json files.

eval_testset_metrics.py rewrites summary.json / metrics.md from only the datasets of that
invocation, so evaluating datasets in separate runs leaves a partial report. Each dataset
folder keeps its own metrics.json ({'summary': ..., 'per_trial': ...}), so the full report
can be regenerated without re-running inference or touching the model.

Usage (run with the posetail env, e.g. via pixi):
    python scripts/merge_eval_summaries.py RUN_DIR [--datasets ds1 ds2 ...] [--out OUT_DIR]

By default every subfolder of RUN_DIR with a metrics.json is merged and the report is written
back into RUN_DIR. Pass --out to write somewhere else and leave RUN_DIR untouched.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_testset_metrics import write_report  # noqa: E402


def cams_used(run_dir, ds):
    """Counter of the camera-name tuples actually used, read from each saved prediction .npz."""
    counts = collections.Counter()
    for f in glob.glob(os.path.join(run_dir, ds, '*.npz')):
        with np.load(f, allow_pickle=True) as d:
            key = tuple(str(c) for c in d['cam_names_used']) if 'cam_names_used' in d.files else None
        counts[key] += 1
    return counts


def cams_label(counts):
    if not counts:
        return 'unknown (no predictions found)'
    if len(counts) == 1:
        (key,) = counts
        return ' '.join(key) if key else 'unknown'
    n = sorted({len(k) for k in counts if k})
    return f'varies per trial ({len(counts)} different sets of {"/".join(map(str, n))} cams)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--datasets', nargs='+', default=None,
                    help='datasets to include, in report order (default: all folders with metrics.json)')
    ap.add_argument('--out', default=None, help='output dir (default: run_dir)')
    args = ap.parse_args()

    datasets = args.datasets or sorted(
        d for d in os.listdir(args.run_dir)
        if not d.startswith('_') and os.path.exists(os.path.join(args.run_dir, d, 'metrics.json')))

    all_summary = {}
    cam_labels = {}
    for ds in datasets:
        path = os.path.join(args.run_dir, ds, 'metrics.json')
        if not os.path.exists(path):
            print(f'skipping {ds}: no {path}')
            continue
        with open(path) as f:
            all_summary[ds] = json.load(f)['summary']
        counts = cams_used(args.run_dir, ds)
        cam_labels[ds] = cams_label(counts)
        if len(counts) == 1 and None not in counts:
            all_summary[ds]['cam_names_used'] = list(next(iter(counts)))
        elif counts:
            all_summary[ds]['cam_names_used'] = 'varies'

    out = args.out or args.run_dir
    os.makedirs(out, exist_ok=True)
    write_report(all_summary, out)

    # write_report's config table prints the dataset defaults (n_views / view_seed), which is
    # not what ran when --cam-names was given; replace it with the cameras read from the outputs
    md_path = os.path.join(out, 'metrics.md')
    with open(md_path) as f:
        md = f.read()
    rows = ['| dataset | cameras used |', '|---|---|'] + [f'| {ds} | {lab} |' for ds, lab in cam_labels.items()]
    md = re.sub(r'## Camera sampling config\n\n.*?\n\n(?=## )',
                '## Cameras used\n\nRead from `cam_names_used` in each saved prediction.\n\n' + '\n'.join(rows) + '\n\n',
                md, count=1, flags=re.S)
    with open(md_path, 'w') as f:
        f.write(md)
    print(f'wrote {os.path.join(out, "summary.json")} and metrics.md for: {list(all_summary)}')


if __name__ == '__main__':
    main()
