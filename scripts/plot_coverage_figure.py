#!/usr/bin/env python3
"""
Generate ESM-C coverage figure for the PlantMetBench paper.

Derives per-organism coverage from the PyTorch data files and produces
a two-panel figure:
  A – strip chart: per-organism ESM-C coverage (%), grouped by split
  B – stacked horizontal bars: total protein counts, embedded vs. missing

Output:
  figures/coverage_figure.pdf   (for Overleaf)
  figures/coverage_figure.png   (300 DPI preview)

Usage:
    python scripts/plot_coverage_figure.py
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO    = Path(__file__).parent.parent
DATA    = REPO / "data"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
C = dict(
    train   = "#3B70A3",
    val     = "#C2782E",
    test    = "#3D8A5C",
    ath     = "#B52B2B",
    missing = "#C4CDD6",
    grid    = "#E3E8EE",
    text    = "#1A1E2A",
    sub     = "#5A6070",
)

# ── Data ───────────────────────────────────────────────────────────────────────
def load_coverage():
    print("Loading data files …")
    data     = torch.load(DATA / "heterodata.pt", weights_only=False)
    nodes_df = pd.read_csv(DATA / "nodes.tsv", sep="\t", low_memory=False)
    prot_emb = torch.load(DATA / "embeddings_protein.pt", weights_only=False)
    splits   = torch.load(DATA / "splits_taxa.pt", weights_only=False)

    # Build pool mask: non-zero L2 norm (matches dataset.py logic)
    prot_ids = nodes_df.loc[nodes_df["node_type"] == "Protein", "node_id"].tolist()
    emb_dim  = next(iter(prot_emb.values())).shape[0]
    feat     = torch.zeros(len(prot_ids), emb_dim)
    for i, nid in enumerate(prot_ids):
        if nid in prot_emb:
            feat[i] = prot_emb[nid]
    pool_mask = feat.norm(dim=-1) > 0

    org_ids     = nodes_df.loc[nodes_df["node_type"] == "Organism", "node_id"].tolist()
    prot_org_ei = data[("Protein", "organism", "Organism")].edge_index

    train_prots = set(splits["train_edge_index"][0].tolist())
    val_prots   = set(splits["val_edge_index"][0].tolist())
    test_prots  = set(splits["test_edge_index"][0].tolist())

    rows = []
    for org_idx, org_iri in enumerate(org_ids):
        p_idxs = prot_org_ei[0][prot_org_ei[1] == org_idx].tolist()
        if not p_idxs:
            continue
        pset = set(p_idxs)
        if   pset & val_prots:   sp = "val"
        elif pset & test_prots:  sp = "test"
        elif pset & train_prots: sp = "train"
        else:                    continue
        n_prot = len(p_idxs)
        n_emb  = int(sum(pool_mask[p].item() for p in p_idxs))
        taxid  = org_iri.split("_")[-1] if "_" in org_iri else org_iri
        rows.append(dict(taxid=taxid, n_prot=n_prot, n_emb=n_emb,
                         coverage=n_emb / n_prot, split=sp))

    for sp in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == sp]
        covs = [r["coverage"] for r in sub]
        print(f"  {sp:6s}: {len(sub):3d} organisms  "
              f"median={np.median(covs):.1%}  "
              f"n_zero={sum(1 for c in covs if c == 0)}")
    return rows


def dot_size(n_prot):
    if n_prot >= 5000:
        return 72
    return max(5.0, min(4.0 + math.sqrt(n_prot / 8) * 2.8, 38.0))


def seeded_jitter(n, half_width, seed):
    rng = np.random.default_rng(seed)
    return (rng.random(n) - 0.5) * 2 * half_width


# ── Figure ─────────────────────────────────────────────────────────────────────
def make_figure(rows):
    by_split = {"train": [], "val": [], "test": []}
    ath = None
    for r in rows:
        by_split[r["split"]].append(r)
        if r["taxid"] == "3702":
            ath = r

    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":        7.5,
        "axes.linewidth":   0.65,
        "xtick.major.size": 2.5,
        "xtick.major.width":0.55,
        "ytick.major.size": 2.5,
        "ytick.major.width":0.55,
        "pdf.fonttype":     42,   # embed fonts in PDF
        "ps.fonttype":      42,
    })

    fig = plt.figure(figsize=(6.8, 3.1))
    gs  = gridspec.GridSpec(
        1, 2, width_ratios=[2.7, 1.1],
        left=0.09, right=0.97, bottom=0.22, top=0.93, wspace=0.45,
    )
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # ── Panel A: strip chart ──────────────────────────────────────────────────
    col_x   = {"train": 0.0, "val": 1.55, "test": 2.9}
    col_col = {"train": C["train"], "val": C["val"], "test": C["test"]}

    for split, orgs in by_split.items():
        cx  = col_x[split]
        col = col_col[split]

        zeros    = [o for o in orgs if o["coverage"] == 0.0]
        nonzeros = [o for o in orgs if o["coverage"] > 0 and o["taxid"] != "3702"]

        # Zero-coverage dense cluster
        if zeros:
            n  = len(zeros)
            xs = cx + seeded_jitter(n, half_width=0.30, seed=10 + ord(split[0]))
            ys =       seeded_jitter(n, half_width=0.007, seed=20 + ord(split[0]))
            ax_a.scatter(xs, ys, s=4.5, color=col, alpha=0.28,
                         linewidths=0, zorder=2)

        # Non-zero organisms
        if nonzeros:
            n  = len(nonzeros)
            ys = np.array([o["coverage"] for o in nonzeros])

            if split == "train":
                # 135 non-zero training orgs: tiny uniform dots, wide jitter
                xs  = cx + seeded_jitter(n, half_width=0.40, seed=30 + ord(split[0]))
                ys  = np.clip(ys + seeded_jitter(n, half_width=0.022, seed=40 + ord(split[0])), 0.001, 1.04)
                ax_a.scatter(xs, ys, s=3.5, color=col, alpha=0.45,
                             linewidths=0, zorder=3)
            else:
                # Val (11) and test (69): sized dots, show individual organisms
                xs  = cx + seeded_jitter(n, half_width=0.27, seed=30 + ord(split[0]))
                ys  = np.clip(ys + seeded_jitter(n, half_width=0.020, seed=40 + ord(split[0])), 0.001, 1.04)
                szs = np.array([dot_size(o["n_prot"]) for o in nonzeros])
                ax_a.scatter(xs, ys, s=szs, color=col, alpha=0.68,
                             linewidths=0.4, edgecolors="white", zorder=3)

    # A. thaliana dot
    if ath:
        ax_a.scatter([col_x["train"]], [ath["coverage"]],
                     s=78, color=C["ath"], alpha=0.92, zorder=6,
                     linewidths=0.9, edgecolors="white")
        ax_a.annotate(
            r"$\it{A.\,thaliana}$" + f"\n{ath['coverage']:.1%}  ({ath['n_prot']:,} prot.)",
            xy=(col_x["train"], ath["coverage"]),
            xytext=(col_x["train"] + 0.52, ath["coverage"] - 0.11),
            fontsize=6.2, color=C["ath"], ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color=C["ath"],
                            lw=0.7, shrinkA=5, shrinkB=2),
        )

    # Median lines
    medians = {s: float(np.median([o["coverage"] for o in orgs])) for s, orgs in by_split.items()}
    for split, med in medians.items():
        cx  = col_x[split]
        col = col_col[split]
        lw  = 1.9
        if med < 0.03:
            # a median at (or near) 0 sits on the zero-coverage cluster; draw the line just above
            ax_a.hlines(0.0, cx - 0.22, cx + 0.22,
                        colors=col, linewidth=lw, zorder=5, alpha=0.95)
            ax_a.text(cx + 0.26, 0.025, f"median {med*100:.0f}%",
                      va="bottom", fontsize=6.0, color=col, fontweight="medium")
        else:
            ax_a.hlines(med, cx - 0.22, cx + 0.22,
                        colors=col, linewidth=lw, zorder=5, alpha=0.95)
            ax_a.text(cx + 0.26, med, f"median {med*100:.0f}%",
                      va="center", fontsize=6.0, color=col, fontweight="medium")

    # Zero-coverage annotation
    n_zero  = sum(1 for o in by_split["train"] if o["coverage"] == 0.0)
    n_train = len(by_split["train"])
    ax_a.annotate(
        f"n={n_zero} ({n_zero/n_train:.0%})\nzero coverage",
        xy=(col_x["train"] - 0.28, 0.005),
        xytext=(col_x["train"] - 0.80, 0.13),
        fontsize=6.0, color=C["train"], ha="center", va="bottom",
        arrowprops=dict(arrowstyle="->", color=C["train"],
                        lw=0.6, shrinkA=2, shrinkB=2),
    )

    # Axes
    ax_a.set_ylim(-0.065, 1.12)
    ax_a.set_xlim(-0.70, 3.70)
    ax_a.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax_a.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_a.set_xticks([col_x[s] for s in ("train", "val", "test")])
    ax_a.set_xticklabels(
        [f"Training\n(n={len(by_split['train'])})", f"Validation\n(n={len(by_split['val'])})", f"Test\n(n={len(by_split['test'])})"],
        fontsize=7.5,
    )
    for tick, sp in zip(ax_a.get_xticklabels(), ("train", "val", "test")):
        tick.set_color(col_col[sp])
    ax_a.set_ylabel("ESM-C coverage (% of species proteins)", fontsize=7.5)
    ax_a.yaxis.grid(True, linewidth=0.4, linestyle="--",
                    color=C["grid"], zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.spines["bottom"].set_color("#D0D8E0")
    ax_a.spines["left"].set_color("#D0D8E0")
    ax_a.tick_params(colors=C["sub"], length=2.5)
    ax_a.text(-0.13, 1.05, "A", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold", color=C["text"])

    # ── Panel B: 100% normalised stacked bars ────────────────────────────────
    # Normalise each split to 100% so val/test are as readable as training.
    bars = [
        # displayed bottom-to-top: test, val, train
        dict(split="test",  y=0, total=1337, ath_emb=0,    other_emb=1002, miss=335),
        dict(split="val",   y=1, total=1260, ath_emb=0,    other_emb=826,  miss=434),
        dict(split="train", y=2, total=11100, ath_emb=5289, other_emb=6621-5289, miss=11100-6621),
    ]
    bar_h = 0.52

    for bd in bars:
        y   = bd["y"]
        col = col_col[bd["split"]]
        tot = bd["total"]
        x0  = 0.0

        if bd["ath_emb"] > 0:
            w = bd["ath_emb"] / tot * 100
            ax_b.barh(y, w, left=x0, height=bar_h,
                      color=C["ath"], alpha=0.87, linewidth=0)
            # Label inside Ath segment
            ax_b.text(x0 + w / 2, y, f"{w:.0f}%",
                      ha="center", va="center", fontsize=5.8,
                      color="white", fontweight="bold")
            x0 += w

        w_emb = bd["other_emb"] / tot * 100
        ax_b.barh(y, w_emb, left=x0, height=bar_h,
                  color=col, alpha=0.80, linewidth=0)
        x0 += w_emb

        w_miss = bd["miss"] / tot * 100
        ax_b.barh(y, w_miss, left=x0, height=bar_h,
                  color=C["missing"], alpha=0.88, linewidth=0)

        # Total ESM-C % label at right
        pct_emb = (bd["ath_emb"] + bd["other_emb"]) / tot * 100
        ax_b.text(101, y, f"{pct_emb:.0f}%\nESM-C",
                  va="center", ha="left", fontsize=6.0, color=C["sub"],
                  linespacing=1.2)

    # A. thaliana label above its segment in the training bar
    ath_pct_mid = (5289 / 11100 * 100) / 2
    ax_b.annotate(
        r"$\it{A.\,thal.}$",
        xy=(ath_pct_mid, 2 + bar_h / 2),
        xytext=(ath_pct_mid, 2 + bar_h / 2 + 0.42),
        fontsize=5.5, color=C["ath"], ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", color=C["ath"], lw=0.5),
    )

    ax_b.set_xlim(0, 125)
    ax_b.set_ylim(-0.6, 2.9)
    ax_b.set_yticks([0, 1, 2])
    ax_b.set_yticklabels(["Test", "Val", "Training"], fontsize=7.5)
    for tick, sp in zip(ax_b.get_yticklabels(), ("test", "val", "train")):
        tick.set_color(col_col[sp])
    ax_b.set_xlabel("% of split proteins", fontsize=7.5)
    ax_b.set_xticks([0, 25, 50, 75, 100])
    ax_b.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=6.5)
    ax_b.xaxis.grid(True, linewidth=0.4, linestyle="--",
                    color=C["grid"], zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.spines["left"].set_color("#D0D8E0")
    ax_b.spines["bottom"].set_color("#D0D8E0")
    ax_b.tick_params(colors=C["sub"], length=2.5)
    ax_b.set_title("ESM-C coverage breakdown", fontsize=7.5, color=C["sub"], pad=5)
    ax_b.text(-0.25, 1.05, "B", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold", color=C["text"])

    # Legend
    legend_items = [
        mpatches.Patch(color=C["ath"],     alpha=0.87, label=r"$\it{A.\,thaliana}$ ESM-C"),
        mpatches.Patch(color=C["train"],   alpha=0.80, label="Other ESM-C"),
        mpatches.Patch(color=C["missing"], alpha=0.88, label="No embedding"),
    ]
    ax_b.legend(handles=legend_items, fontsize=5.8, loc="lower center",
                bbox_to_anchor=(0.5, -0.52),
                ncol=1, frameon=True, framealpha=0.92, edgecolor="#D8DDE4",
                handlelength=0.9, handleheight=0.8,
                borderpad=0.5, labelspacing=0.25)

    # ── Save ──────────────────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"coverage_figure.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")

    return fig


if __name__ == "__main__":
    rows = load_coverage()
    make_figure(rows)
