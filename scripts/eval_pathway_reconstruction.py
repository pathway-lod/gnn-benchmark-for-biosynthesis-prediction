#!/usr/bin/env python3
"""
Pathway reconstruction analysis for the A. thaliana baseline (ath_pathway split).

For each test reaction, computes the rank of the true catalyst in the protein pool.
Then groups reactions by pathway membership and reports how many pathways are fully,
partially, or not reconstructed by the model.

Usage (from repo root):
    python scripts/eval_pathway_reconstruction.py [--seeds 42 0 1] [--no-plot]

Output:
    figures/pathway_reconstruction.pdf / .png
    results/pathway_reconstruction.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "training"))

from dataset import load_data
from models import build_model
from metrics import _score_all

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = REPO / "data"
RUNS_DIR  = REPO / "runs" / "ath_sage" / "sage_L2_h128_dot_lr0.0001"
FIG_DIR   = REPO / "figures"
RES_DIR   = REPO / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

TARGET_EDGE = ("Protein", "catalyzes", "Interaction")


# ── Per-reaction ranking ───────────────────────────────────────────────────────
@torch.no_grad()
def per_reaction_ranks(gnn, predictor, ctx):
    """Return (interaction_idx, rank) arrays for all test reactions.

    rank is 1-indexed; a reaction whose true catalyst is not in the pool gets
    rank=pool_size+1 (guaranteed miss, excluded from pathway stats).
    """
    gnn.eval()
    test_data = ctx.test_data

    z     = gnn(test_data.x_dict, test_data.edge_index_dict)
    z_src = z[ctx.src_type]    # [n_proteins, dim]
    z_dst = z[ctx.dst_type]    # [n_interactions, dim]

    lbl    = test_data[TARGET_EDGE].edge_label
    pos_ei = test_data[TARGET_EDGE].edge_label_index[:, lbl == 1]
    n_total = pos_ei.shape[1]

    prot_mask = ctx.embedded_protein_mask
    pool_idx  = torch.where(prot_mask)[0]
    pool_embs = z_src[pool_idx]          # [n_pool, dim]
    n_pool    = pool_embs.shape[0]

    # Which test edges have an in-pool catalyst?
    found    = torch.searchsorted(pool_idx, pos_ei[0])
    in_pool  = (
        (found < pool_idx.shape[0]) &
        (pool_idx[found.clamp(max=pool_idx.shape[0]-1)] == pos_ei[0])
    )
    pos_src_pool = torch.where(in_pool, found, torch.full_like(found, -1))

    # Score all pool proteins against all test reactions: [n_pool, n_total]
    dst_embs   = z_dst[pos_ei[1]]                           # [n_total, dim]
    all_scores = _score_all(predictor, pool_embs, dst_embs) # [n_pool, n_total]

    ranks = torch.full((n_total,), n_pool + 1, dtype=torch.long)
    valid = torch.where(in_pool)[0]
    if valid.numel() > 0:
        rows        = pos_src_pool[valid]
        true_scores = all_scores[rows, valid]
        ranks[valid] = (all_scores[:, valid] >= true_scores.unsqueeze(0)).sum(dim=0)

    inter_idx  = pos_ei[1].cpu().numpy()   # interaction node indices
    ranks_np   = ranks.cpu().numpy()
    in_pool_np = in_pool.cpu().numpy()

    return inter_idx, ranks_np, in_pool_np, n_pool


# ── Pathway mapping ────────────────────────────────────────────────────────────
def build_pathway_map(heterodata):
    """Return {interaction_idx: [pathway_idx, ...]} from the KG."""
    ei = heterodata[("Interaction", "is_part_of", "Pathway")].edge_index
    mapping = defaultdict(list)
    for inter, pway in zip(ei[0].tolist(), ei[1].tolist()):
        mapping[inter].append(pway)
    return mapping


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print("Loading dataset …")
    ctx = load_data(
        data_dir=DATA_DIR,
        split_type="ath_pathway",
        species_pool=True,
        download=True,
        print_summary=False,
    )

    print("Loading KG for pathway structure …")
    heterodata = torch.load(DATA_DIR / "heterodata.pt", weights_only=False)
    pathway_map = build_pathway_map(heterodata)
    print(f"  Reaction→pathway entries: {sum(len(v) for v in pathway_map.values()):,}")

    # ── Collect per-reaction ranks across seeds ────────────────────────────────
    all_ranks_by_inter = defaultdict(list)   # inter_idx → [rank_seed1, ...]
    in_pool_by_inter   = {}

    for seed in args.seeds:
        ckpt_dir = RUNS_DIR / f"seed_{seed}"
        ckpt_files = sorted(ckpt_dir.glob("*best*.pt"))
        if not ckpt_files:
            print(f"  [skip] seed {seed}: no checkpoint found in {ckpt_dir}")
            continue
        ckpt_path = ckpt_files[0]
        print(f"\nSeed {seed}: loading {ckpt_path.name}")

        model = build_model(
            ctx, gnn_name="sage", hidden_dim=128, num_layers=2,
            decoder="dot", dropout=0.3, random_seed=seed,
        )
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        inter_idx, ranks, in_pool, n_pool = per_reaction_ranks(model.gnn, model.predictor, ctx)
        print(f"  Pool size: {n_pool}  |  Test reactions: {len(ranks)}  "
              f"|  In-pool: {in_pool.sum()}")

        for ii, rk, ip in zip(inter_idx, ranks, in_pool):
            all_ranks_by_inter[int(ii)].append(int(rk))
            in_pool_by_inter[int(ii)] = bool(ip)

    if not all_ranks_by_inter:
        print("No seed results found. Exiting.")
        return

    n_seeds = len(args.seeds)

    # ── Aggregate: mean rank per reaction ─────────────────────────────────────
    reactions = {}   # inter_idx → {mean_rank, hit_at_k, in_pool}
    for ii, rk_list in all_ranks_by_inter.items():
        mean_rk = float(np.mean(rk_list))
        hit     = float(np.mean([r <= args.k for r in rk_list]))  # mean across seeds
        reactions[ii] = {
            "mean_rank": mean_rk,
            "hit_at_k":  hit,
            "in_pool":   in_pool_by_inter[ii],
            "n_seeds":   len(rk_list),
        }

    # Filter to in-pool reactions only (matches P-H@50 denominator)
    eval_reactions = {ii: v for ii, v in reactions.items() if v["in_pool"]}
    print(f"\nEvaluable reactions (catalyst in pool): {len(eval_reactions)}")

    # ── Pathway-level reconstruction ─────────────────────────────────────────
    # pathway_idx → list of (inter_idx, hit_at_k)
    pathway_results = defaultdict(list)
    for ii, info in eval_reactions.items():
        for pway in pathway_map.get(ii, []):
            pathway_results[pway].append(info["hit_at_k"])

    n_pathways_with_test = len(pathway_results)
    print(f"Pathways with ≥1 evaluable test reaction: {n_pathways_with_test}")

    # Per-pathway reconstruction rate = mean hit@k across its test reactions
    pw_rates = {pw: float(np.mean(hits)) for pw, hits in pathway_results.items()}
    pw_sizes = {pw: len(hits)            for pw, hits in pathway_results.items()}

    rates = np.array(list(pw_rates.values()))
    sizes = np.array([pw_sizes[pw] for pw in pw_rates])

    # Summary
    thresholds = [1.0, 0.75, 0.5, 0.25, 0.0]
    print(f"\n── Pathway reconstruction @ top-{args.k} ──────────────────────")
    print(f"  {'Category':<30}  {'n pathways':>12}  {'%':>6}")
    print(f"  {'Fully reconstructed (100%)' :<30}  {(rates == 1.0).sum():>12}  "
          f"{(rates == 1.0).mean()*100:>5.1f}%")
    print(f"  {'≥75% reactions correct'    :<30}  {(rates >= 0.75).sum():>12}  "
          f"{(rates >= 0.75).mean()*100:>5.1f}%")
    print(f"  {'≥50% reactions correct'    :<30}  {(rates >= 0.50).sum():>12}  "
          f"{(rates >= 0.50).mean()*100:>5.1f}%")
    print(f"  {'≥25% reactions correct'    :<30}  {(rates >= 0.25).sum():>12}  "
          f"{(rates >= 0.25).mean()*100:>5.1f}%")
    print(f"  {'Not reconstructed (0%)'    :<30}  {(rates == 0.0).sum():>12}  "
          f"{(rates == 0.0).mean()*100:>5.1f}%")
    print(f"\n  Mean pathway reconstruction rate: {rates.mean()*100:.1f}%")
    print(f"  Median:                           {np.median(rates)*100:.1f}%")
    print(f"  (Based on {n_pathways_with_test} pathways, {len(eval_reactions)} reactions, "
          f"{n_seeds} seeds)")

    # Reactions per pathway in test set
    print(f"\n── Pathway test-set sizes ────────────────────────────────────")
    print(f"  Median reactions per pathway in test: {np.median(sizes):.0f}")
    print(f"  Mean:  {sizes.mean():.1f}   Max: {sizes.max()}   "
          f"Single-reaction pathways: {(sizes == 1).sum()}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_json = RES_DIR / "pathway_reconstruction.json"
    out_data = {
        "k": args.k,
        "seeds": args.seeds,
        "n_pathways_with_test_reactions": n_pathways_with_test,
        "n_evaluable_reactions": len(eval_reactions),
        "mean_pathway_reconstruction_rate": float(rates.mean()),
        "median_pathway_reconstruction_rate": float(np.median(rates)),
        "fully_reconstructed":   int((rates == 1.0).sum()),
        "at_least_75pct":        int((rates >= 0.75).sum()),
        "at_least_50pct":        int((rates >= 0.50).sum()),
        "at_least_25pct":        int((rates >= 0.25).sum()),
        "not_reconstructed":     int((rates == 0.0).sum()),
        "per_pathway": {
            str(pw): {"rate": float(r), "n_test_reactions": int(pw_sizes[pw])}
            for pw, r in pw_rates.items()
        },
    }
    out_json.write_text(json.dumps(out_data, indent=2))
    print(f"\nSaved → {out_json}")

    if args.no_plot:
        return

    # ── Figure ────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.linewidth": 0.65,
        "xtick.major.size": 2.5, "xtick.major.width": 0.55,
        "ytick.major.size": 2.5, "ytick.major.width": 0.55,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    C = dict(train="#3B70A3", full="#3D8A5C", partial="#C2782E",
             none="#B8C5CE", text="#1A1E2A", sub="#5A6070", grid="#E3E8EE")

    fig = plt.figure(figsize=(6.8, 3.2))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1.0],
                            left=0.10, right=0.97, bottom=0.28, top=0.95, wspace=0.44)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # Panel A: histogram of per-pathway reconstruction rates
    bins = np.linspace(0, 1, 21)
    col_map = lambda r: C["full"] if r == 1.0 else (C["none"] if r == 0.0 else C["partial"])
    colors_hist = [C["none"], C["partial"], C["partial"], C["partial"],
                   C["partial"], C["partial"], C["partial"], C["partial"],
                   C["partial"], C["partial"], C["partial"], C["partial"],
                   C["partial"], C["partial"], C["partial"], C["partial"],
                   C["partial"], C["partial"], C["partial"], C["full"]]

    counts, _, patches = ax_a.hist(rates, bins=bins, edgecolor="white",
                                    linewidth=0.5, color=C["partial"], alpha=0.82)
    # recolor the 0% and 100% bars
    patches[0].set_facecolor(C["none"])
    patches[-1].set_facecolor(C["full"])
    patches[0].set_alpha(0.82)
    patches[-1].set_alpha(0.88)

    ax_a.set_xlabel(f"Pathway reconstruction rate (P-H@{args.k})", fontsize=7.5)
    ax_a.set_ylabel("Number of pathways", fontsize=7.5)
    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_a.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_a.yaxis.grid(True, linewidth=0.4, linestyle="--", color=C["grid"], zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.spines["bottom"].set_color("#D0D8E0")
    ax_a.spines["left"].set_color("#D0D8E0")
    ax_a.tick_params(colors=C["sub"], length=2.5)

    # Annotate key bars with headroom above the tallest bar
    n_full = int((rates == 1.0).sum())
    n_none = int((rates == 0.0).sum())
    ax_a.set_ylim(0, max(n_none, counts.max()) * 1.25)
    ax_a.text(0.0, n_none * 1.04, f"n={n_none} ({n_none/len(rates)*100:.0f}%)",
              ha="center", va="bottom", fontsize=6.2, color=C["none"], fontweight="medium")
    ax_a.text(1.0, n_full * 1.04, f"n={n_full} ({n_full/len(rates)*100:.0f}%)",
              ha="center", va="bottom", fontsize=6.2, color=C["full"], fontweight="medium")
    ax_a.text(-0.02, 1.03, "A", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold", color=C["text"])

    # Panel B: stacked bar chart — fraction of pathways in each category
    # stratified by pathway size (1, 2–5, 6+ test reactions)
    size_bins = [(1,1,"1"), (2,5,"2–5"), (6,999,"≥6")]
    cat_labels = ["Fully\nreconstructed", "Partial\n(1–99%)", "Not\nreconstructed"]
    cat_colors = [C["full"], C["partial"], C["none"]]
    cat_alphas  = [0.88, 0.82, 0.82]

    bar_data = []
    xtick_labels = []
    for lo, hi, lbl in size_bins:
        mask = (sizes >= lo) & (sizes <= hi)
        sub  = rates[mask]
        if len(sub) == 0:
            continue
        n_full_s = (sub == 1.0).sum()
        n_part_s = ((sub > 0) & (sub < 1.0)).sum()
        n_none_s = (sub == 0.0).sum()
        bar_data.append([n_full_s/len(sub), n_part_s/len(sub), n_none_s/len(sub)])
        xtick_labels.append(f"{lbl} rxn\n(n={mask.sum()})")

    bar_data = np.array(bar_data)
    x = np.arange(len(bar_data))
    bottoms = np.zeros(len(bar_data))
    for i, (cat, col, alpha) in enumerate(zip(cat_labels, cat_colors, cat_alphas)):
        ax_b.bar(x, bar_data[:, i], bottom=bottoms, color=col, alpha=alpha,
                 width=0.55, linewidth=0, label=cat)
        bottoms += bar_data[:, i]

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(xtick_labels, fontsize=7)
    ax_b.set_ylabel("Fraction of pathways", fontsize=7.5)
    ax_b.set_ylim(0, 1.08)
    ax_b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_b.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_b.yaxis.grid(True, linewidth=0.4, linestyle="--", color=C["grid"], zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.spines["bottom"].set_color("#D0D8E0")
    ax_b.spines["left"].set_color("#D0D8E0")
    ax_b.tick_params(colors=C["sub"], length=2.5)
    ax_b.set_title("By pathway size (test rxns/pathway)", fontsize=7, color=C["sub"], pad=4)
    ax_b.legend(fontsize=6, loc="lower center",
                bbox_to_anchor=(0.5, -0.58),
                ncol=1, frameon=True, framealpha=0.9, edgecolor="#D8DDE4",
                handlelength=0.8, handleheight=0.8,
                borderpad=0.4, labelspacing=0.2)
    ax_b.text(-0.22, 1.03, "B", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold", color=C["text"])

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"pathway_reconstruction.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
