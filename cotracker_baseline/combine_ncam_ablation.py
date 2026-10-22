"""
Combine per-nviews summaries into a single ncam_ablation_summary.csv.

Supports two source formats:
  - combined_summary_metrics.csv  (cotracker / mvtracker pipeline)
  - summary.json                  (posetail eval_testset_metrics.py)

For summary.json the output columns are unprefixed (avg_jaccard, mte, …)
using mean aggregates, matching the notebook convention:
    df['avg_jaccard']   df['n_views']

Usage:
    python combine_ncam_ablation.py <ablation_root>

Writes <ablation_root>/ncam_ablation_summary.csv with an n_views column prepended.
"""

import json
import sys
import re
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "cotracker-baseline/cotracker-outputs-ncam-ablation-dexycb"

rows = []
for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    m = re.match(r"nviews-(\d+)$", d.name)
    if not m:
        continue
    n = int(m.group(1))

    csv = d / "combined_summary_metrics.csv"
    summary = d / "summary.json"

    if csv.exists():
        df = pd.read_csv(csv)
        df.insert(0, "n_views", n)
        rows.extend(df.to_dict("records"))
    elif summary.exists():
        data = json.loads(summary.read_text())
        for dataset, metrics in data.items():
            row = {"n_views": n, "dataset": dataset}
            for key, val in metrics.items():
                if key.endswith("_mean"):
                    row[key[: -len("_mean")]] = val
            rows.append(row)
    else:
        print(f"  [skip] no combined_summary_metrics.csv or summary.json in {d.name}")

if not rows:
    print("No data found.")
    sys.exit(1)

out = pd.DataFrame(rows).sort_values("n_views").reset_index(drop=True)
out_path = root / "ncam_ablation_summary.csv"
out.to_csv(out_path, index=False)
print(f"Wrote {len(out)} row(s) → {out_path}")
print(out.to_string(index=False))
