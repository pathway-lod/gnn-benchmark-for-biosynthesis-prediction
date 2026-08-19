#!/usr/bin/env python3
"""Run the baseline model over multiple random seeds and report mean ± std.

Trains one model per seed, saving each run independently, then aggregates the
final results across seeds to report the mean and standard deviation for every
metric. The taxa split is fixed; only model initialization and negative sampling
vary across seeds.

Usage:
    python train_seeds.py                          # 5 seeds, default args
    python train_seeds.py --seeds 42,0,1,2,3       # explicit seed list
    python train_seeds.py --hidden-dim 256 --num-layers 3  # larger model

All other train.py flags are accepted and passed through to each seed run.
Summary is saved to runs/<run-name>/seeds_summary.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RUNS_DIR = Path(__file__).parent.parent / "runs"

TRACKED_METRICS = [
    "val_ph50", "val_ph10", "val_ph1", "val_ph5",
    "val_cp_auc", "val_cp_ap",
    "test_ph50", "test_ph10", "test_ph1", "test_ph5",
    "test_cp_auc", "test_cp_ap",
    "best_epoch",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seeds", type=str, default="42,0,1,2,3",
                    help="Comma-separated list of random seeds (default: 42,0,1,2,3)")
    ap.add_argument("--run-name", type=str, default="baseline",
                    help="Base name; each seed runs as <run-name>/seed_<s>")
    # Allow any other train.py args to pass through
    args, remaining = ap.parse_known_args()
    return args, remaining


def run_seed(seed: int, run_name: str, extra_args: list[str]) -> Path:
    """Launch a single-seed training subprocess. Returns run_dir."""
    seed_run_name = f"{run_name}/seed_{seed}"
    cmd = [
        sys.executable, str(Path(__file__).parent / "train.py"),
        "--seed", str(seed),
        "--run-name", seed_run_name,
    ] + extra_args

    print(f"\n{'='*70}")
    print(f"  Seed {seed}  →  runs/{seed_run_name}/")
    print(f"{'='*70}")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [WARNING] Seed {seed} exited with code {result.returncode}")

    return RUNS_DIR / seed_run_name


def aggregate(seed_dirs: list[Path]) -> dict:
    """Read report.json from each seed dir and compute mean ± std."""
    import statistics

    per_seed: dict[str, list[float]] = {m: [] for m in TRACKED_METRICS}

    for d in seed_dirs:
        report_path = d / "report.json"
        if not report_path.exists():
            print(f"  [WARNING] No report.json in {d} — skipping")
            continue
        report = json.loads(report_path.read_text())
        res    = report.get("results", {})
        for m in TRACKED_METRICS:
            v = res.get(m)
            if v is not None:
                per_seed[m].append(float(v))

    summary: dict = {}
    for m, vals in per_seed.items():
        if not vals:
            continue
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[m] = {"mean": round(mean, 6), "std": round(std, 6), "n": len(vals), "values": vals}

    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "="*70)
    print("  Multi-seed summary")
    print("="*70)
    header = f"  {'Metric':<20}  {'Mean':>8}  {'Std':>8}  n"
    print(header)
    print("  " + "-"*50)
    for m in TRACKED_METRICS:
        if m not in summary:
            continue
        s = summary[m]
        print(f"  {m:<20}  {s['mean']:>8.4f}  {s['std']:>8.4f}  {s['n']}")
    print()


def main():
    args, extra_args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print(f"PlantMetBench — multi-seed training")
    print(f"  Seeds    : {seeds}")
    print(f"  Run name : {args.run_name}")
    if extra_args:
        print(f"  Extra    : {' '.join(extra_args)}")

    seed_dirs = []
    for seed in seeds:
        seed_dir = run_seed(seed, args.run_name, extra_args)
        seed_dirs.append(seed_dir)

    summary = aggregate(seed_dirs)
    print_summary(summary)

    out_dir = RUNS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "seeds_summary.json"
    out_path.write_text(json.dumps({
        "seeds": seeds,
        "run_name": args.run_name,
        "extra_args": extra_args,
        "metrics": summary,
    }, indent=2))
    print(f"Summary saved → {out_path}")


if __name__ == "__main__":
    main()
