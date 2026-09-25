#!/usr/bin/env python3
"""Hidden-dimension ablation for the best model.

Compares HeteroSAGE+Res&Jump (residual_jumping_sage, 2 layers, bidirectional)
across hidden_dim in {128, 256, 512, 1024}, 5 seeds each.

Reads <runs-dir>/residual_jumping_sage_L2_h*_dot_lr0.0001/seed_*/report_*.json,
computes the mean and standard deviation over seeds of test P-H@50, P-H@10 and
P-H@5 for each hidden_dim, and draws one grouped bar chart (metrics on the
x-axis, one bar per hidden_dim).

Usage:
    python plot_hidden_dim_ablation.py --runs-dir ../runs/<run-name> --out-dir ../figures
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS  = ["test_ph50", "test_ph10", "test_ph5"]
METRIC_LABELS = {"test_ph50": "P-H@50", "test_ph10": "P-H@10", "test_ph5": "P-H@5"}
COLORS  = ["#AEC6CF", "#B5EAD7", "#FFDAC1", "#E2CFC4"]  # pastel blue/green/orange/mauve
HATCHES = ["", "///", "xxx", "..."]


def load_summary(base_dir: Path) -> dict[int, dict[str, dict]]:
    exp_dirs = sorted(
        base_dir.glob("residual_jumping_sage_L2_h*_dot_lr0.0001"),
        key=lambda p: int(re.search(r"_h(\d+)_", p.name).group(1)),
    )
    if not exp_dirs:
        raise SystemExit(f"No residual_jumping_sage_L2_h*_dot_lr0.0001 folders under {base_dir}")

    summary: dict[int, dict[str, dict]] = {}
    for d in exp_dirs:
        print(f"Processing {d.name}")
        h = int(re.search(r"_h(\d+)_", d.name).group(1))
        per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
        for seed_dir in sorted(d.glob("seed_*")):
            reports = sorted(seed_dir.glob("report_*.json"))
            if not reports:
                print(f"  [WARNING] no report_*.json under {seed_dir} -- skipping")
                continue
            res = json.loads(reports[-1].read_text())["results"]
            for m in METRICS:
                per_metric[m].append(float(res[m]))
        summary[h] = {
            m: {
                "mean": statistics.mean(v),
                "std": statistics.stdev(v) if len(v) > 1 else 0.0,
                "n": len(v),
            }
            for m, v in per_metric.items()
        }
    return summary


def plot(summary: dict[int, dict[str, dict]], save_path: Path) -> None:
    hidden_dims = sorted(summary)
    n_groups = len(METRICS)
    n_bars   = len(hidden_dims)
    width    = 0.8 / n_bars
    x        = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, h in enumerate(hidden_dims):
        means = [summary[h][m]["mean"] * 100 for m in METRICS]
        stds  = [summary[h][m]["std"] * 100 for m in METRICS]
        offset = (i - (n_bars - 1) / 2) * width
        bars = ax.bar(x + offset, means, width=width * 0.92, yerr=stds, capsize=3,
                      color=COLORS[i % len(COLORS)], hatch=HATCHES[i % len(HATCHES)],
                      edgecolor="#555555", linewidth=0.9, label=f"h={h}", zorder=3)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 1.2,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel("Test set score (%)", fontsize=12)
    ax.set_title("HeteroSAGE+Res&Jump (L2, bidirectional): hidden-dimension ablation", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=n_bars, fontsize=10)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Plot saved -> {save_path.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=Path(__file__).parent.parent / "runs",
                    help="Directory holding the residual_jumping_sage_L2_h*_dot_lr0.0001 run folders")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent.parent / "figures",
                    help="Where the summary JSON and the plot are written")
    args = ap.parse_args()
    OUT_DIR = args.out_dir

    summary = load_summary(args.runs_dir)
    for h, s in summary.items():
        print(f"h={h}")
        for m, v in s.items():
            print(f"  {m}: {v['mean']:.4f} +/- {v['std']:.4f}  (n={v['n']})")

    out_json = OUT_DIR / "hidden_dim_ablation_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved -> {out_json}")

    plot(summary, OUT_DIR / "hidden_dim_ablation")


if __name__ == "__main__":
    main()
