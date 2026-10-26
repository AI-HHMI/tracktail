"""Collect the bench.py jsonl files of a results folder into summary.csv and summary.md (fps and peak GPU memory
per method x views x keypoints).

Usage: python summarize.py RESULTS_DIR
"""
import json
import sys
from pathlib import Path

import pandas as pd

METHOD_ORDER = ["tracktail", "cotracker", "mvtracker-gt", "mvtracker-dust3r", "arrow"]


def to_markdown(df):
    rows = [[str(c) for c in df.columns]] + [[str(v) for v in r] for r in df.itertuples(index=False)]
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    fmt = lambda r: "| " + " | ".join(x.ljust(n) for x, n in zip(r, w)) + " |"
    return "\n".join([fmt(rows[0]), "|" + "|".join("-" * (n + 2) for n in w) + "|"] + [fmt(r) for r in rows[1:]])


def main():
    out = Path(sys.argv[1])
    recs = [json.loads(l) for f in sorted(out.glob("*.jsonl")) for l in open(f) if l.strip()]
    if not recs:
        print(f"no results in {out}")
        return
    df = pd.DataFrame(recs)
    # the last record wins if a configuration was run twice
    df = df.drop_duplicates(["method", "views", "kpts"], keep="last")
    df["method"] = pd.Categorical(df["method"], [m for m in METHOD_ORDER if m in set(df["method"])] +
                                  sorted(set(df["method"]) - set(METHOD_ORDER)))
    df = df.sort_values(["method", "views", "kpts"])
    cols = [c for c in ["method", "views", "kpts", "view_note", "n_frames", "status", "fps", "sec", "peak_alloc_gb",
                        "peak_reserved_gb", "weights_gb", "stage_fps", "dust3r_fps", "dust3r_peak_alloc_gb", "gpu",
                        "host"] if c in df]
    df[cols].to_csv(out / "summary.csv", index=False)

    def cell(r, key, fmt):
        return fmt.format(r[key]) if r["status"] == "ok" else r["status"].upper()

    lines = []
    gpus = ", ".join(sorted(set(df["gpu"])))
    lines.append(f"# Speed benchmark ({gpus}, {df['n_frames'].iloc[0]} frames, Point Odyssey)\n")
    notes = sorted({r.get("view_note") for r in recs if r.get("view_note")})
    if notes:
        lines.append("Note: " + "; ".join(notes) + ". This is a duplicated input slot, not an additional independent camera.\n")
    for title, key, fmt in [("Overall frames per second", "fps", "{:.2f}"),
                            ("Peak GPU memory allocated (GB)", "peak_alloc_gb", "{:.2f}"),
                            ("Peak GPU memory reserved (GB)", "peak_reserved_gb", "{:.2f}")]:
        lines.append(f"## {title}\n")
        tab = df.assign(v=[cell(r, key, fmt) for _, r in df.iterrows()],
                        col=[f"{v} views" for v in df["views"]])
        piv = tab.pivot_table(index=["method", "kpts"], columns="col", values="v", aggfunc="first", observed=True)
        piv = piv[sorted(piv.columns, key=lambda c: int(c.split()[0]))]
        lines.append(to_markdown(piv.reset_index()) + "\n")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
