#!/usr/bin/env python3
"""Render main-paper and appendix results tables from seeds_summary.json files.

Reads results/<run_name>/seeds_summary.json for each model variant and prints:
  1. Main paper table  — P-H@50 (val+test), CP-AUC (test); mean ± std
  2. Appendix table    — full metrics (P-H@1/5/10/50, CP-AUC, CP-AP); val + test

Usage (from repo root):
    python scripts/render_results_tables.py

Add or reorder MODEL_RUNS to change which models appear.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# ── Model registry ─────────────────────────────────────────────────────────────
# (display_name, results_subdir)
MODEL_RUNS: list[tuple[str, str]] = [
    ("HeteroSAGE (1 layer)",  "sage_1layer"),
    ("HeteroSAGE (2 layers)", "sage_2layer"),
    ("HeteroSAGE (3 layers)", "sage_3layer"),
]

RANDOM_BASELINE = {
    "val_ph1":  0.0001, "val_ph5":  0.0006, "val_ph10": 0.0012, "val_ph50": 0.0059,
    "test_ph1": 0.0001, "test_ph5": 0.0006, "test_ph10": 0.0012, "test_ph50": 0.0059,
    "val_cp_auc": 0.500, "val_cp_ap": 0.500,
    "test_cp_auc": 0.500, "test_cp_ap": 0.500,
}


def load(run_name: str) -> dict | None:
    path = RESULTS_DIR / run_name / "seeds_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(summary: dict, metric: str, pct: bool = True) -> str:
    """Format mean ± std from a seeds_summary metrics dict."""
    if metric not in summary["metrics"]:
        return r"\TODO{--}"
    m = summary["metrics"][metric]
    mean, std = m["mean"], m["std"]
    if pct:
        return rf"{mean*100:.1f} $\pm$ {std*100:.1f}\%"
    else:
        return rf"{mean:.3f} $\pm$ {std:.3f}"


def fmt_random(val: float, pct: bool = True) -> str:
    if pct:
        return rf"{val*100:.2f}\%"
    return rf"{val:.3f}"


# ── Table 1: Main paper ────────────────────────────────────────────────────────

def print_main_table(summaries: list[tuple[str, dict | None]]) -> None:
    print("% ── MAIN PAPER TABLE ─────────────────────────────────────────────────────")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{HeteroSAGE with varying depth on PlantMetBench (mean\,$\pm$\,std, 5 seeds).")
    print(r"P-H@50: fraction of reactions where the true catalyst is in the top 50")
    print(r"of 8{,}445 embedded proteins. Random P-H@50 $\approx$ 0.59\%.}")
    print(r"\label{tab:main_results}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Model & \multicolumn{2}{c}{P-H@50} & \multicolumn{2}{c}{CP-AUC} & Best \\")
    print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    print(r" & Val & Test & Val & Test & epoch \\")
    print(r"\midrule")
    print(r"Random baseline & 0.59\% & 0.59\% & 0.500 & 0.500 & — \\")
    print(r"\midrule")
    for name, summ in summaries:
        if summ is None:
            print(rf"{name} & \TODO{{--}} & \TODO{{--}} & \TODO{{--}} & \TODO{{--}} & \TODO{{--}} \\")
        else:
            vp50  = fmt(summ, "val_ph50")
            tp50  = fmt(summ, "test_ph50")
            vauc  = fmt(summ, "val_cp_auc", pct=False)
            tauc  = fmt(summ, "test_cp_auc", pct=False)
            epoch = fmt(summ, "best_epoch", pct=False)
            print(rf"{name} & {vp50} & {tp50} & {vauc} & {tauc} & {epoch} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


# ── Table 2: Appendix ──────────────────────────────────────────────────────────

def print_appendix_table(summaries: list[tuple[str, dict | None]]) -> None:
    print("% ── APPENDIX FULL TABLE ──────────────────────────────────────────────────")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Full evaluation metrics for all model variants")
    print(r"(mean\,$\pm$\,std across 5 seeds: 42, 0, 1, 2, 3).}")
    print(r"\label{tab:full_results}")
    print(r"\begin{tabular}{llcccccc}")
    print(r"\toprule")
    print(r"Model & Split & P-H@1 & P-H@5 & P-H@10 & P-H@50 & CP-AUC & CP-AP \\")
    print(r"\midrule")

    # Random baseline row
    r = RANDOM_BASELINE
    print(rf"Random & Val  "
          rf"& {fmt_random(r['val_ph1'])} & {fmt_random(r['val_ph5'])} "
          rf"& {fmt_random(r['val_ph10'])} & {fmt_random(r['val_ph50'])} "
          rf"& {fmt_random(r['val_cp_auc'],False)} & {fmt_random(r['val_cp_ap'],False)} \\")
    print(r"\midrule")

    for name, summ in summaries:
        for split in ("val", "test"):
            label = r"\multirow{2}{*}{" + name + r"}" if split == "val" else ""
            sname = "Val" if split == "val" else "Test"
            if summ is None:
                row = r" & \TODO{--}" * 6
            else:
                row = (
                    f" & {fmt(summ, f'{split}_ph1')}"
                    f" & {fmt(summ, f'{split}_ph5')}"
                    f" & {fmt(summ, f'{split}_ph10')}"
                    f" & {fmt(summ, f'{split}_ph50')}"
                    f" & {fmt(summ, f'{split}_cp_auc', pct=False)}"
                    f" & {fmt(summ, f'{split}_cp_ap', pct=False)}"
                )
            print(rf"{label} & {sname}{row} \\")
        print(r"\midrule")

    # remove last \midrule and replace with \bottomrule — done below
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    summaries = [(name, load(run)) for name, run in MODEL_RUNS]

    missing = [name for name, s in summaries if s is None]
    if missing:
        print(f"% NOTE: results not yet available for: {', '.join(missing)}")
        print(f"% Run: python training/train_seeds.py --run-name <name> --num-layers <n>")
        print()

    print_main_table(summaries)
    print_appendix_table(summaries)


if __name__ == "__main__":
    main()
