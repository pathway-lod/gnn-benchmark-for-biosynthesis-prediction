#!/usr/bin/env python3
"""Data-scaling ablation: train the same model on increasing fractions of the
training positives and check whether val/test performance is still rising at
100% of the data, or has flattened.

Usage:
    python train_frac_ablation.py
    python train_frac_ablation.py --fracs 0.1,0.25,0.5,0.75,1.0 --seeds 42,0,1
    python train_frac_ablation.py --run-name frac-ablation --epochs 650
    python train_frac_ablation.py --run-name frac-ablation --plot-only   # re-draw from a saved summary

Any other argument is passed through to train.py. Runs are written under
runs/<run-name>/, and frac_ablation_summary.json and frac_ablation.png/.pdf
are written to <out-dir>/<run-name>/ (default: runs/<run-name>/).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

RUNS_DIR = Path(__file__).parent.parent / "runs"   # where train.py writes its runs
OUT_DIR  = RUNS_DIR

TRACKED_METRICS = [
    "val_ph50", "val_ph10", "val_cp_auc", "val_cp_ap",
    "test_ph50", "test_ph10", "test_cp_auc", "test_cp_ap",
    "best_epoch",
]

# Best configuration: HeteroSAGE+Res&Jump, 2 layers, hidden dimension 128, bidirectional
DEFAULT_EXTRA_ARGS = [
    "--gnn-name", "residual_jumping_sage",
    "--num-layers", "2",
    "--hidden-dim", "128",
    "--bidirectional",
    "--epochs", "650",
]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--fracs", type=str, default="0.1,0.25,0.5,0.75,1.0",
                    help="Comma-separated training-data fractions (default: 0.1,0.25,0.5,0.75,1.0)")
    ap.add_argument("--seeds", type=str, default="42,0,1",
                    help="Comma-separated seeds per fraction (default: 42,0,1)")
    ap.add_argument("--run-name", type=str, default="frac-ablation",
                    help="Base name; each run lands under <run-name>/frac_<f>/<config_tag>/seed_<s>")
    ap.add_argument("--plot-only", action="store_true",
                    help="Do not train; re-draw the plot from <out-dir>/<run-name>/frac_ablation_summary.json")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="Directory for the summary JSON and the plot (default: runs/)")
    # Allow any other train.py args to pass through (overrides DEFAULT_EXTRA_ARGS on conflict)
    args, remaining = ap.parse_known_args()
    return args, remaining


def _extract_flag(extra_args: list[str], name: str, default: str) -> str:
    for i, tok in enumerate(extra_args):
        if tok == f"--{name}" and i + 1 < len(extra_args):
            return extra_args[i + 1]
        if tok.startswith(f"--{name}="):
            return tok.split("=", 1)[1]
    return default


def merge_args(extra_args: list[str]) -> list[str]:
    """DEFAULT_EXTRA_ARGS, with any flag the user passed on the CLI overriding it."""
    user_flags = {tok.lstrip("-").split("=")[0] for tok in extra_args if tok.startswith("--")}
    merged = list(extra_args)
    i = 0
    while i < len(DEFAULT_EXTRA_ARGS):
        flag = DEFAULT_EXTRA_ARGS[i].lstrip("-")
        is_bool_flag = i + 1 >= len(DEFAULT_EXTRA_ARGS) or DEFAULT_EXTRA_ARGS[i + 1].startswith("--")
        if flag not in user_flags:
            merged.append(DEFAULT_EXTRA_ARGS[i])
            if not is_bool_flag:
                merged.append(DEFAULT_EXTRA_ARGS[i + 1])
        i += 1 if is_bool_flag else 2
    return merged


def run_one(frac: float, seed: int, run_name: str, extra_args: list[str], config_tag: str) -> Path:
    frac_tag = f"frac_{frac:g}"
    cmd = [
        sys.executable, str(Path(__file__).parent / "train.py"),
        "--seed", str(seed),
        "--train-frac", str(frac),
        "--run-name", f"{run_name}/{frac_tag}",
        "--no-dataset-summary",
    ] + extra_args

    run_dir = RUNS_DIR / run_name / frac_tag / config_tag / f"seed_{seed}"
    print(f"\n{'='*70}\n  train_frac={frac}  seed={seed}  ->  {run_dir}/\n{'='*70}")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [WARNING] frac={frac} seed={seed} exited with code {result.returncode}")
    return run_dir


def aggregate(run_dirs: dict[float, list[Path]]) -> dict:
    summary: dict[float, dict] = {}
    for frac, dirs in run_dirs.items():
        per_seed: dict[str, list[float]] = {m: [] for m in TRACKED_METRICS}
        for d in dirs:
            report_paths = sorted(d.rglob("report_*.json"))
            if not report_paths:
                print(f"  [WARNING] No report_*.json under {d} -- skipping")
                continue
            res = json.loads(report_paths[-1].read_text()).get("results", {})
            for m in TRACKED_METRICS:
                v = res.get(m)
                if v is not None:
                    per_seed[m].append(float(v))

        frac_summary = {}
        for m, vals in per_seed.items():
            if not vals:
                continue
            mean = statistics.mean(vals)
            std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
            frac_summary[m] = {"mean": round(mean, 6), "std": round(std, 6), "n": len(vals), "values": vals}
        summary[frac] = frac_summary
    return summary


def plot(summary: dict, save_path: Path) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    fracs = sorted(summary)
    pcts = [f * 100 for f in fracs]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3))

    for ax, key, ylabel in [
        (axes[0], "ph50", "P-H@50 (%)"),
        (axes[1], "cp_auc", "CP-AUC"),
    ]:
        scale = 100 if key == "ph50" else 1
        for split, color, marker in [("val", "#0072B2", "*"), ("test", "#CC79A7", "o")]:
            metric = f"{split}_{key}"
            means = [summary[f].get(metric, {}).get("mean") for f in fracs]
            stds  = [summary[f].get(metric, {}).get("std", 0.0) for f in fracs]
            if any(m is None for m in means):
                continue
            means = np.array(means) * scale
            stds  = np.array(stds) * scale
            ax.plot(pcts, means, marker=marker, markersize=6,
                    linewidth=2.2, color=color, label=split, zorder=3)
            ax.fill_between(pcts, means - stds, means + stds,
                            color=color, alpha=0.2, linewidth=0, zorder=2)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)

    fig.supxlabel("Training positives used (%)", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Plot saved -> {save_path.with_suffix('.png')}")


def main():
    args, extra_args = parse_args()
    out_dir = args.out_dir / args.run_name

    if args.plot_only:
        data = json.loads((out_dir / "frac_ablation_summary.json").read_text())
        plot({float(k): v for k, v in data["summary"].items()}, out_dir / "frac_ablation")
        return

    fracs = sorted(float(f.strip()) for f in args.fracs.split(","))
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    merged_args = merge_args(extra_args)

    gnn_name   = _extract_flag(merged_args, "gnn-name", "sage")
    num_layers = _extract_flag(merged_args, "num-layers", "2")
    hidden_dim = _extract_flag(merged_args, "hidden-dim", "128")
    decoder    = _extract_flag(merged_args, "decoder", "dot")
    lr         = str(float(_extract_flag(merged_args, "lr", "1e-4")))
    config_tag = f"{gnn_name}_L{num_layers}_h{hidden_dim}_{decoder}_lr{lr}"

    print("PlantMetBench -- data-scaling ablation")
    print(f"  Fracs    : {fracs}")
    print(f"  Seeds    : {seeds}")
    print(f"  Config   : {config_tag}")
    print(f"  Extra    : {' '.join(merged_args)}")

    run_dirs: dict[float, list[Path]] = {f: [] for f in fracs}
    for frac in fracs:
        for seed in seeds:
            run_dirs[frac].append(run_one(frac, seed, args.run_name, merged_args, config_tag))

    summary = aggregate(run_dirs)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frac_ablation_summary.json"
    out_path.write_text(json.dumps({
        "fracs": fracs, "seeds": seeds, "config_tag": config_tag,
        "extra_args": merged_args, "summary": summary,
    }, indent=2))
    print(f"\nSummary saved -> {out_path}")

    print(f"\n{'Frac':>6}  {'val_ph50':>10}  {'test_ph50':>10}  {'val_cp_auc':>10}  {'test_cp_auc':>11}")
    for f in fracs:
        s = summary.get(f, {})
        def fmt(m):
            e = s.get(m)
            return f"{e['mean']:.4f}±{e['std']:.4f}" if e else "n/a"
        print(f"{f:>6.2f}  {fmt('val_ph50'):>10}  {fmt('test_ph50'):>10}  "
              f"{fmt('val_cp_auc'):>10}  {fmt('test_cp_auc'):>11}")

    plot(summary, out_dir / "frac_ablation")


if __name__ == "__main__":
    main()
