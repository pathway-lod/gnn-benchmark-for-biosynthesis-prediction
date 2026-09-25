#!/usr/bin/env python3
"""
Reaction-level detail behind the "What makes a pathway fully reconstructable?"
appendix analysis: which specific reactions underlie the fully-reconstructed
pathways, their known-catalyst (isoenzyme) counts, and their EC top-level
class -- reusing eval_pathway_reconstruction.py's ranking logic so the
"fully reconstructed" set matches its headline numbers exactly.

Usage (from repo root):
    python scripts/eval_pathway_recon_detail.py \
        --runs-dir runs/ath_resjump/residual_jumping_sage_L2_h128_dot_lr0.0001 \
        --seeds 42 0 1 2 3 --organism-taxid 3702
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "scripts"))

from build_clean_ctx import interaction_node_ids, load_ctx_for_run
from models import build_model
from eval_pathway_reconstruction import per_reaction_ranks, build_pathway_map

DATA_DIR = REPO / "data"
TARGET_EDGE = ("Protein", "catalyzes", "Interaction")

EC_NAMES = {1: "Oxidoreductases", 2: "Transferases", 3: "Hydrolases", 4: "Lyases",
            5: "Isomerases", 6: "Ligases", 7: "Translocases"}


def top_ec(ec):
    try:
        return int(str(ec).split(".")[0])
    except (ValueError, AttributeError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1, 2, 3])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--organism-taxid", default=None,
                     help="If set, restrict 'known catalysts' count to this "
                          "organism's proteins (e.g. 3702 for A. thaliana). "
                          "If unset, counts catalysts across all organisms.")
    args = ap.parse_args()

    print("Loading KG for pathway structure and catalyst counts …")
    heterodata = torch.load(DATA_DIR / "heterodata.pt", weights_only=False)
    raw_pathway_map = build_pathway_map(heterodata)
    nodes = pd.read_csv(DATA_DIR / "nodes.tsv", sep="\t", low_memory=False)

    # ── known-catalyst count per reaction (raw Interaction index) ──────────────
    cat_ei = heterodata[TARGET_EDGE].edge_index
    if args.organism_taxid:
        po_ei = heterodata[("Protein", "organism", "Organism")].edge_index
        org_nodes = nodes[nodes.node_type == "Organism"].reset_index(drop=True)
        target_org_idx = {i for i, row in org_nodes.iterrows()
                           if f"NCBITaxon_{args.organism_taxid}" in row["node_id"]}
        prot_to_org = {}
        for p, o in zip(po_ei[0].tolist(), po_ei[1].tolist()):
            prot_to_org.setdefault(p, o)
        allowed_prots = {p for p, o in prot_to_org.items() if o in target_org_idx}
        catalyst_count = defaultdict(int)
        for p, i in zip(cat_ei[0].tolist(), cat_ei[1].tolist()):
            if p in allowed_prots:
                catalyst_count[i] += 1
    else:
        catalyst_count = defaultdict(int)
        for i in cat_ei[1].tolist():
            catalyst_count[i] += 1

    raw_inter_nodes = nodes[nodes.node_type == "Interaction"].reset_index(drop=True)
    raw_id_to_idx = {nid: i for i, nid in enumerate(raw_inter_nodes["node_id"])}
    ec_by_nodeid = dict(zip(raw_inter_nodes["node_id"], raw_inter_nodes["ec_number"]))

    # ── Collect per-reaction ranks across seeds (mirrors eval_pathway_reconstruction.py) ──
    all_ranks_by_inter = defaultdict(list)
    in_pool_by_inter = {}
    pathway_map = None

    for seed in args.seeds:
        ckpt_dir = args.runs_dir / f"seed_{seed}"
        ckpt_files = sorted(ckpt_dir.glob("*best*.pt"))
        if not ckpt_files:
            print(f"  [skip] seed {seed}: no checkpoint found in {ckpt_dir}")
            continue
        ckpt_path = ckpt_files[0]
        print(f"Seed {seed}: loading {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, weights_only=False)
        hp = ckpt.get("hparams", {})

        ctx = load_ctx_for_run(hp, DATA_DIR, seed=seed, download=False, print_summary=False)
        if pathway_map is None:
            raw_ids = interaction_node_ids(ctx.nodes_df, "raw")
            pathway_map = {raw_ids[i]: pw for i, pw in raw_pathway_map.items()}
        inter_ids = interaction_node_ids(ctx.nodes_df, hp.get("data_type", "raw"))

        model = build_model(
            ctx, gnn_name=hp.get("gnn_name", "sage"), hidden_dim=hp.get("hidden_dim", 128),
            num_layers=hp.get("num_layers", 2), decoder=hp.get("decoder", "dot"),
            dropout=hp.get("dropout", 0.3), random_seed=seed,
            use_norm=hp.get("layer_norm", False),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        inter_idx, ranks, in_pool, n_pool = per_reaction_ranks(model.gnn, model.predictor, ctx)
        for ii, rk, ip in zip(inter_idx, ranks, in_pool):
            nid = inter_ids[int(ii)]
            all_ranks_by_inter[nid].append(int(rk))
            in_pool_by_inter[nid] = bool(ip)

    n_seeds = len(args.seeds)
    reactions = {}
    for nid, rk_list in all_ranks_by_inter.items():
        hit = float(np.mean([r <= args.k for r in rk_list]))
        reactions[nid] = {"hit_at_k": hit, "in_pool": in_pool_by_inter[nid]}

    eval_reactions = {nid: v for nid, v in reactions.items() if v["in_pool"]}
    print(f"\nEvaluable reactions: {len(eval_reactions)}")

    pathway_results = defaultdict(list)   # pway -> [(nid, hit_at_k), ...]
    for nid, info in eval_reactions.items():
        for pway in pathway_map.get(nid, []):
            pathway_results[pway].append((nid, info["hit_at_k"]))

    pw_rate = {pw: float(np.mean([h for _, h in items])) for pw, items in pathway_results.items()}
    fully_pathways = [pw for pw, r in pw_rate.items() if r == 1.0]
    print(f"Fully reconstructed pathways: {len(fully_pathways)}")

    fully_reactions = set()
    for pw in fully_pathways:
        for nid, h in pathway_results[pw]:
            fully_reactions.add(nid)
    print(f"Distinct reactions underlying them: {len(fully_reactions)}")

    # recurrence check
    reaction_pathway_count = defaultdict(int)
    for pw in fully_pathways:
        for nid, h in pathway_results[pw]:
            reaction_pathway_count[nid] += 1
    recurring = {nid: c for nid, c in reaction_pathway_count.items() if c > 1}
    print(f"Reactions recurring across >1 fully-reconstructed pathway: {len(recurring)}")

    # ── isoenzyme (known-catalyst) counts ───────────────────────────────────────
    def get_count(nid):
        idx = raw_id_to_idx.get(nid)
        return catalyst_count.get(idx, 0) if idx is not None else 0

    fully_counts = np.array([get_count(nid) for nid in fully_reactions])
    all_counts   = np.array([get_count(nid) for nid in eval_reactions])

    print(f"\n── Isoenzyme (known-catalyst) counts ──────────────────────")
    print(f"  Fully-reconstructed reactions (n={len(fully_counts)}): "
          f"median={np.median(fully_counts):.0f}  mean={fully_counts.mean():.2f}  max={fully_counts.max()}")
    print(f"  All evaluable reactions       (n={len(all_counts)}): "
          f"median={np.median(all_counts):.0f}  mean={all_counts.mean():.2f}  max={all_counts.max()}")

    # ── EC class breakdown ───────────────────────────────────────────────────────
    def ec_class(nid):
        return top_ec(ec_by_nodeid.get(nid))

    all_ec = pd.Series([ec_class(nid) for nid in eval_reactions])
    fully_ec = pd.Series([ec_class(nid) for nid in fully_reactions])

    print(f"\n── EC top-level class breakdown ────────────────────────────")
    print(f"  {'EC':<18} {'total evaluable':>16} {'fully recon.':>13} {'%':>7}")
    for ec, name in sorted(EC_NAMES.items()):
        tot = int((all_ec == ec).sum())
        full = int((fully_ec == ec).sum())
        pct = full / tot * 100 if tot else float("nan")
        print(f"  EC{ec} {name:<14} {tot:>16} {full:>13} {pct:>6.1f}%")
    tot_none = int(all_ec.isna().sum())
    full_none = int(fully_ec.isna().sum())
    print(f"  {'(no EC)':<18} {tot_none:>16} {full_none:>13} "
          f"{(full_none/tot_none*100 if tot_none else float('nan')):>6.1f}%")

    # ── most granular EC codes among fully-reconstructed reactions ─────────────
    fully_ec_full = pd.Series([ec_by_nodeid.get(nid) for nid in fully_reactions]).dropna()
    print(f"\n── Most common full EC codes among fully-reconstructed reactions ──")
    print(fully_ec_full.value_counts().head(10))


if __name__ == "__main__":
    main()
