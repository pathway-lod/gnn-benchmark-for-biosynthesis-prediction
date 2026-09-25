#!/usr/bin/env python3
"""Model-size scatter: accuracy vs. convergence speed vs. size.

One point per model and depth: x = mean best epoch, y = mean test P-H@50
(both across seeds), marker area proportional to the number of trainable
parameters.

Two sources of runs are combined:
  - per-seed runs: every h128 folder under <runs-dir> (test P-H@50 and best epoch
    from report_*.json, trainable parameters from "Trainable parameters: N" in
    each seed's train.log);
  - pre-aggregated summaries: <summary-dir>/seeds_summary_*.json, which carry
    metrics.test_ph50 / metrics.best_epoch (mean, std), "gnn_name",
    "num_layers" and a "parameters" field. Add the parameter count to these files
    by hand if it is missing (it is printed in the train.log of each run).

Usage:
    python plot_model_scatter.py --runs-dir ../runs/<run-name> --out-dir ../figures
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl



# One color + hatch per architecture family, so L1/L2/L3 of the same GNN read
# as a group and stay distinguishable in grayscale / for colorblind readers.
# Same pastel palette as plot_hidden_dim_ablation.py, for consistency.
FAMILY_COLORS = {
    "sage":                  "#6C9EB1",  # pastel blue
    "residual_sage":         "#B5EAD7",  # pastel mint
    "residual_jumping_sage": "#FFDAC1",  # pastel peach
    "mix":                   "#E38854",  # pastel mauve
    "gat":                   "#5A76E7",  # pastel lavender
    "hgt":                   "#FFB7B2",  # pastel salmon
    "rgcn":                  "#FFF1B9",  # pastel butter
}
FAMILY_HATCHES = {
    "sage":                  "",
    "residual_sage":         "///",
    "residual_jumping_sage": "xxx",
    "mix":                   "...",
    "gat":                   "+++",
    "hgt":                   "\\\\\\",
    "rgcn":                  "ooo",
}

NAMES = {
    "sage":                  "HeteroSAGE",
    "residual_sage":         "HeteroSAGE+Res",
    "residual_jumping_sage": "HeteroSAGE+Res&Jump",
    "mix":                   "MIX",
    "gat":                   "GAT",
    "hgt":                   "HGT",
    "rgcn":                  "R-GCN",
}

PARAM_RE = re.compile(r"Trainable parameters:\s*([\d,]+)")


def load_experiments(base_dir: Path) -> list[dict]:
    exp_dirs = sorted(base_dir.glob("*_h128_*"))
    if not exp_dirs:
        return []

    experiments = []
    for d in exp_dirs:
        m = re.match(r"(.+)_L(\d+)_h128_.+", d.name)
        if not m:
            print(f"  [WARNING] could not parse folder name {d.name} -- skipping")
            continue
        gnn_name, num_layers = m.group(1), int(m.group(2))

        test_ph50s, best_epochs, n_params_list = [], [], []
        for seed_dir in sorted(d.glob("seed_*")):
            reports = sorted(seed_dir.glob("report_*.json"))
            if not reports:
                print(f"  [WARNING] no report_*.json under {seed_dir} -- skipping")
                continue
            res = json.loads(reports[-1].read_text())["results"]
            test_ph50s.append(float(res["test_ph50"]))
            best_epochs.append(float(res["best_epoch"]))

            log_path = seed_dir / "train.log"
            match = PARAM_RE.search(log_path.read_text()) if log_path.exists() else None
            if match:
                n_params_list.append(int(match.group(1).replace(",", "")))

        if not test_ph50s:
            continue

        experiments.append({
            "label": f"{gnn_name}_L{num_layers}",
            "gnn_name": gnn_name,
            "num_layers": num_layers,
            "mean_test_ph50": statistics.mean(test_ph50s),
            "std_test_ph50": statistics.stdev(test_ph50s) if len(test_ph50s) > 1 else 0.0,
            "mean_best_epoch": statistics.mean(best_epochs),
            "std_best_epoch": statistics.stdev(best_epochs) if len(best_epochs) > 1 else 0.0,
            "n_params": round(statistics.mean(n_params_list)) if n_params_list else None,
            "n_seeds": len(test_ph50s),
        })
    return experiments


def load_json_experiments(json_dir: Path) -> list[dict]:
    """Load pre-aggregated experiments from <json_dir>/seeds_summary_*.json, which
    carry metrics.{test_ph50,best_epoch}.{mean,std} and a "parameters" field,
    keyed by "gnn_name" (rather than seed-by-seed reports)."""
    if not json_dir.is_dir():
        return []

    experiments = []
    for f in sorted(json_dir.glob("seeds_summary_*.json")):
        data = json.loads(f.read_text())
        gnn_name = data["gnn_name"]
        num_layers = int(data["num_layers"])
        metrics = data["metrics"]

        n_params = None
        if data.get("parameters"):
            n_params = int(str(data["parameters"]).replace(",", ""))

        experiments.append({
            "label": f"{gnn_name}_L{num_layers}",
            "gnn_name": gnn_name,
            "num_layers": num_layers,
            "mean_test_ph50": metrics["test_ph50"]["mean"],
            "std_test_ph50": metrics["test_ph50"]["std"],
            "mean_best_epoch": metrics["best_epoch"]["mean"],
            "std_best_epoch": metrics["best_epoch"]["std"],
            "n_params": n_params,
            "n_seeds": metrics["test_ph50"]["n"],
        })
    return experiments


def load_all_experiments(base_dir: Path, json_dir: Path) -> list[dict]:
    experiments = load_experiments(base_dir) + load_json_experiments(json_dir)
    if not experiments:
        raise SystemExit(f"No experiments found under {base_dir} or {json_dir}")
    return experiments


def plot(experiments: list[dict], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    n_params = [e["n_params"] for e in experiments if e["n_params"]]
    lo, hi = min(n_params), max(n_params)

    def area(n_params: int) -> float:
        # sqrt scaling so *area* (not radius) tracks parameter count; clipped to a readable range
        frac = (n_params - lo) / (hi - lo) if hi > lo else 0.5
        return 120 + frac * 1600

    def round_nice(n: int) -> int:
        # round to 2 significant figures, e.g. 1,075,840 -> 1,100,000
        magnitude = 10 ** (len(str(n)) - 2)
        return round(n / magnitude) * magnitude

    for e in experiments:
        color = FAMILY_COLORS.get(e["gnn_name"], "#888888")
        hatch = FAMILY_HATCHES.get(e["gnn_name"], "")
        test_ph50_pct = e["mean_test_ph50"] * 100
        ax.scatter(e["mean_best_epoch"], test_ph50_pct, s=area(e["n_params"]),
                  color=color, alpha=0.55, hatch=hatch, edgecolor="#333333",
                  linewidth=1.0, zorder=3)
        ax.annotate(f"L{e['num_layers']}", (e["mean_best_epoch"], test_ph50_pct),
                    ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)

    ax.set_xlabel("Mean best epoch (across seeds)")
    ax.set_ylabel("Mean test P-H@50 (%, across seeds)")
    ax.set_title(r"$\mathbf{Trade\!-\!off\ analysis}$: accuracy vs. convergence speed vs. size")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Fixed-size proxy handles for the architecture legend -- reusing an actual
    # data point's (possibly tiny) marker made small-model swatches look muddy,
    # since the gray edge dominates a small marker's visible area.
    families_present = [f for f in FAMILY_COLORS if f in {e["gnn_name"] for e in experiments}]
    family_handles = [
        ax.scatter([], [], s=200, color=FAMILY_COLORS[f], alpha=0.55,
                  hatch=FAMILY_HATCHES[f], edgecolor="#6B5151", linewidth=1.0,
                  label=NAMES.get(f, f))
        for f in families_present
    ]
    family_legend = ax.legend(handles=family_handles, title="Architecture",
                              loc="upper left", bbox_to_anchor=(0.2, 1),
                              frameon=True, facecolor="white", edgecolor="white",
                              framealpha=1.0)
    ax.add_artist(family_legend)

    # Separate legend explaining marker size -> parameter count. Built from its
    # own handles (not ax.legend()'s default "everything labeled on the axes")
    # so it doesn't also pick up the architecture handles above.
    size_handles = [
        ax.scatter([], [], s=area(n), color="#D6BEBE", alpha=0.55,
                  edgecolor="#333333", linewidth=1.0, label=f"{round_nice(n):,}")
        for n in (lo, (lo + hi) // 2, hi)
    ]
    ax.legend(handles=size_handles, title="Model size", loc="upper left",
              bbox_to_anchor=(0., 1), frameon=True, facecolor="white",
              edgecolor="white", framealpha=1.0, labelspacing=1.6,
              borderpad=1.2)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Plot saved -> {save_path.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).parent.parent
    ap.add_argument("--runs-dir", type=Path, default=root / "runs",
                    help="Directory holding the <gnn>_L<k>_h128_* run folders")
    ap.add_argument("--summary-dir", type=Path, default=None,
                    help="Directory with seeds_summary_*.json files (default: <runs-dir>/jsons)")
    ap.add_argument("--out-dir", type=Path, default=root / "figures",
                    help="Where the summary JSON and the plot are written")
    args = ap.parse_args()
    OUT_DIR = args.out_dir

    experiments = load_all_experiments(args.runs_dir, args.summary_dir or args.runs_dir / "jsons")
    for e in sorted(experiments, key=lambda e: e["mean_test_ph50"], reverse=True):
        print(f"{e['label']:<28}  test_ph50={e['mean_test_ph50']:.4f}±{e['std_test_ph50']:.4f}  "
              f"best_epoch={e['mean_best_epoch']:.1f}±{e['std_best_epoch']:.1f}  "
              f"params={e['n_params']:,}  n_seeds={e['n_seeds']}")

    out_json = OUT_DIR / "model_scatter_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(experiments, indent=2))
    print(f"Summary saved -> {out_json}")

    plot(experiments, OUT_DIR / "model_scatter")


if __name__ == "__main__":
    main()
