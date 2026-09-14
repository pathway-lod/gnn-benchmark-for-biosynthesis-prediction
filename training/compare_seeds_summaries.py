#!/usr/bin/env python3
"""Compare multiple train_seeds.py runs (different GNN architectures / configs)
side by side, ranked by the primary metric (val P-H@50).

Usage:
    python compare_seeds_summaries.py --run-name baseline
    python compare_seeds_summaries.py --run-name baseline --metric test_ph50

Reads every runs/<run-name>/*/seeds_summary_*.json produced by train_seeds.py
and prints a mean +/- std table, one row per config, sorted best-first on the
chosen metric (default: val_ph50, the primary metric per data/README.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS_DIR = Path(__file__).parent.parent / "runs_clean"

DISPLAY_METRICS = ["val_ph50", "val_ph10", "val_cp_auc", "test_ph50", "test_ph10", "test_cp_auc"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", type=str, default="baseline")
    ap.add_argument("--metric", type=str, default="val_ph50",
                     help="Metric to sort by, best (highest mean) first")
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run_name
    summary_paths = sorted(run_dir.glob("*/seeds_summary_*.json"))
    if not summary_paths:
        raise SystemExit(f"No seeds_summary_*.json found under {run_dir}/*/")

    configs = []
    for p in summary_paths:
        d = json.loads(p.read_text())
        configs.append(d)

    def sort_key(d):
        m = d["metrics"].get(args.metric)
        return -(m["mean"] if m else float("-inf"))

    configs.sort(key=sort_key)

    header = f"{'config_tag':<32}" + "".join(f"{m:>16}" for m in DISPLAY_METRICS)
    print(header)
    print("-" * len(header))
    for d in configs:
        row = f"{d['config_tag']:<32}"
        for m in DISPLAY_METRICS:
            entry = d["metrics"].get(m)
            if entry:
                row += f"{entry['mean']:>8.3f}±{entry['std']:<5.3f}"
            else:
                row += f"{'—':>16}"
        print(row)

    with_metric = [d for d in configs if args.metric in d["metrics"]]
    without_metric = [d["config_tag"] for d in configs if args.metric not in d["metrics"]]
    if without_metric:
        print(f"\n[WARNING] {len(without_metric)} config(s) have no '{args.metric}' data at all "
              f"(every seed run likely crashed before writing a report_*.json -- check its "
              f"train.log): {', '.join(without_metric)}")

    if not with_metric:
        raise SystemExit(f"\nNo config has any '{args.metric}' data -- nothing to rank.")

    best = with_metric[0]
    print(f"\nBest by {args.metric}: {best['config_tag']} "
          f"({best['metrics'][args.metric]['mean']:.3f} ± {best['metrics'][args.metric]['std']:.3f}, "
          f"n={best['metrics'][args.metric]['n']} seeds)")


if __name__ == "__main__":
    main()
