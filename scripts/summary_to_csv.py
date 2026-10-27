"""Write the summary.json of an eval_testset_metrics.py run (rebuilt over all datasets by merge_eval_summaries.py) as a
csv: one row per dataset, one column per entry of its summary (n_trials, <metric>_mean, <metric>_median, ...). Entries
that are lists or dicts (e.g. the cameras used) are left out.

Usage:
    python scripts/summary_to_csv.py RUN_DIR [--out RUN_DIR/summary.csv]
"""
import argparse
import csv
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=None, help='default: summary.csv in run_dir')
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, 'summary.json')) as f:
        summary = json.load(f)
    rows = [{'dataset': ds, **{k: v for k, v in stats.items() if not isinstance(v, (list, dict))}}
            for ds, stats in summary.items() if ds]
    cols = ['dataset'] + sorted({k for r in rows for k in r if k != 'dataset'})

    out = args.out or os.path.join(args.run_dir, 'summary.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out}: {len(rows)} dataset(s), {len(cols) - 1} columns')


if __name__ == '__main__':
    main()
