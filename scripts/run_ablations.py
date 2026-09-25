#!/usr/bin/env python3
"""Run all shortcut-ablation experiments and collect results → results/ablations.json

Each experiment re-introduces exactly one shortcut relative to the full baseline.
Runs train.py four times with different flags, then summarises results side-by-side.

Ablation plan (Table E.1 in the appendix):
  baseline          : all shortcuts removed (default flags)
  keep_pathways     : --keep-pathways            (re-enables is_part_of edges)
  keep_catalyzed_by : --keep-catalyzed-by        (re-enables reverse edge)
  no_disjoint       : --disjoint-train-ratio 0   (re-enables 1-hop shortcut)
  ec_features       : --ec-features              (re-enables EC one-hot features)

Usage:
    cd training/
    python ../scripts/run_ablations.py

    # or from repo root:
    python scripts/run_ablations.py

Extra training flags are forwarded to every run:
    python scripts/run_ablations.py --epochs 100 --seed 0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TRAIN_SCRIPT = Path(__file__).parent.parent / "training" / "train.py"
RUNS_DIR     = Path(__file__).parent.parent / "runs"
RESULTS_DIR  = Path(__file__).parent.parent / "results"

ABLATIONS: list[dict] = [
    {
        "name":        "baseline",
        "description": "All shortcuts removed (default baseline)",
        "extra_flags": [],
    },
    {
        "name":        "keep_pathways",
        "description": "+Pathway co-membership (is_part_of edges kept)",
        "extra_flags": ["--keep-pathways"],
    },
    {
        "name":        "keep_catalyzed_by",
        "description": "+Reverse catalysis edge (catalyzed_by kept)",
        "extra_flags": ["--keep-catalyzed-by"],
    },
    {
        "name":        "no_disjoint",
        "description": "No disjoint train ratio (all train edges in MP graph)",
        "extra_flags": ["--disjoint-train-ratio", "0"],
    },
    {
        "name":        "ec_features",
        "description": "+EC one-hot features on Interaction nodes",
        "extra_flags": ["--ec-features"],
    },
]

REPORTED_METRICS = ["val_ph50", "val_ph10", "val_cp_auc", "test_ph50", "test_ph10", "test_cp_auc"]


def run_one(name: str, extra_flags: list[str], extra_args: list[str]) -> dict:
    run_name = f"ablations/{name}"
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--run-name", run_name,
        *extra_flags,
        *extra_args,
    ]
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=TRAIN_SCRIPT.parent)
    if result.returncode != 0:
        print(f"WARNING: {name} exited with code {result.returncode}")
    report_path = RUNS_DIR / run_name / "report.json"
    if not report_path.exists():
        print(f"WARNING: report.json not found at {report_path}")
        return {}
    with open(report_path) as f:
        report = json.load(f)
    return report.get("results", {})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="Run only these ablation names (e.g. --only baseline ec_features)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip runs that already have a report.json")
    args, extra_args = ap.parse_known_args()

    to_run = ABLATIONS
    if args.only:
        to_run = [a for a in ABLATIONS if a["name"] in args.only]
        if not to_run:
            print(f"No matching ablations for: {args.only}")
            raise SystemExit(1)

    all_results: dict[str, dict] = {}

    for abl in to_run:
        name = abl["name"]
        report_path = RUNS_DIR / "ablations" / name / "report.json"

        if args.skip_existing and report_path.exists():
            print(f"Skipping {name} (report.json already exists)")
            with open(report_path) as f:
                all_results[name] = json.load(f).get("results", {})
            continue

        results = run_one(name, abl["extra_flags"], extra_args)
        all_results[name] = results

    # Summary table
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'Name':<25} {'val_ph50':>9} {'test_ph50':>10} {'val_cp_auc':>11}")
    print("-" * 57)
    for abl in to_run:
        name = abl["name"]
        r    = all_results.get(name, {})
        vp50  = r.get("val_ph50",   float("nan"))
        tp50  = r.get("test_ph50",  float("nan"))
        vauc  = r.get("val_cp_auc", float("nan"))
        print(f"{name:<25} {vp50:>9.4f} {tp50:>10.4f} {vauc:>11.4f}")

    # Write results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "ablations": [
            {
                "name":        abl["name"],
                "description": abl["description"],
                "flags":       abl["extra_flags"],
                "results":     all_results.get(abl["name"], {}),
            }
            for abl in to_run
        ]
    }
    out_path = RESULTS_DIR / "ablations.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
