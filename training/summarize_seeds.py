#!/usr/bin/env python3
"""Aggregate an already-trained multi-seed run into mean ± std per metric.

Unlike train_seeds.py (which launches the training subprocesses itself and
aggregates a fixed TRACKED_METRICS list as it goes), this reads a config
folder that already contains one seed_<s>/ subdir per seed -- each with a
report_*.json written by train.py -- and computes mean/std over every key
found under "results", whatever those happen to be.

Usage:
    python summarize_seeds.py runs/clean-data/mix_L3_h128_dot_lr0.001
    python summarize_seeds.py runs/clean-data/mix_L3_h128_dot_lr0.001 --metric val_ph50

Writes seeds_summary.json into the given folder and prints a table.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_seed_results(exp_dir: Path) -> dict[str, list[float]]:
    """Read results dict from each seed_*/report_*.json under exp_dir.

    Returns {metric_name: [value_per_seed, ...]}, preserving the key order
    of the first report encountered.
    """
    seed_dirs = sorted(exp_dir.glob("seed_*"))
    if not seed_dirs:
        raise SystemExit(f"No seed_* subdirectories found under {exp_dir}")

    per_seed: dict[str, list[float]] = {}
    for d in seed_dirs:
        report_paths = sorted(d.glob("report_*.json"))
        if not report_paths:
            print(f"  [WARNING] No report_*.json under {d} -- skipping")
            continue
        if len(report_paths) > 1:
            print(f"  [WARNING] {len(report_paths)} report_*.json under {d} -- using the last one")
        report = json.loads(report_paths[-1].read_text())
        results = report.get("results", {})
        for k, v in results.items():
            if v is None:
                continue
            per_seed.setdefault(k, []).append(float(v))

    return per_seed


def summarize(per_seed: dict[str, list[float]]) -> dict:
    summary = {}
    for metric, vals in per_seed.items():
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[metric] = {"mean": round(mean, 6), "std": round(std, 6), "n": len(vals), "values": vals}
    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 70)
    print("  Multi-seed summary")
    print("=" * 70)
    print(f"  {'Metric':<20}  {'Mean':>10}  {'Std':>10}  n")
    print("  " + "-" * 55)
    for metric, s in summary.items():
        print(f"  {metric:<20}  {s['mean']:>10.4f}  {s['std']:>10.4f}  {s['n']}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir", type=str,
                     help="Folder for one multi-seed experiment, e.g. "
                          "runs/clean-data/mix_L3_h128_dot_lr0.001")
    ap.add_argument("--metric", type=str, default=None,
                     help="If given, print only this metric's mean ± std (n seeds) after the table")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_dir():
        raise SystemExit(f"Not a directory: {exp_dir}")

    per_seed = load_seed_results(exp_dir)
    if not per_seed:
        raise SystemExit(f"No results found in any report_*.json under {exp_dir}")

    summary = summarize(per_seed)
    print_summary(summary)

    if args.metric:
        s = summary.get(args.metric)
        if s is None:
            print(f"[WARNING] Metric {args.metric!r} not found in results")
        else:
            print(f"{args.metric}: {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']} seeds)")

    out_path = exp_dir / "seeds_summary.json"
    out_path.write_text(json.dumps({
        "exp_dir": str(exp_dir),
        "n_seed_dirs": len(sorted(exp_dir.glob("seed_*"))),
        "metrics": summary,
    }, indent=2))
    print(f"Summary saved -> {out_path}")


if __name__ == "__main__":
    main()
