#!/usr/bin/env python3
"""
Robustness check: how much does a held-out organism's OWN graph neighbourhood
(its other proteins/gene products, linked via is_about/encodes -- never via
Organism nodes, which are removed in the default config) contribute to its
proteins' P-H@50, versus relying only on their own ESM-C feature?

Held-out (val + test) organisms already cannot receive message-passing
information FROM training organisms: Organism nodes are removed in the
default config, and (Protein, is_about, Protein) / (GeneProduct, is_about,
GeneProduct) edges are 99.9% / 95.2% same-organism (verified separately).
This script goes further: it strips EVERY non-target edge touching a
held-out organism's Protein/GeneProduct nodes (including same-organism
edges), so those nodes are scored using only their own 0-hop feature
(propagated through each layer's self-transform, never aggregated from
any neighbour). No retraining -- this re-scores the existing best
checkpoints on a structurally modified val/test graph.

Usage (from repo root):
    python scripts/eval_heldout_isolated.py --seeds 42 0 1 2 3
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "training"))

from build_clean_ctx import load_ctx_for_run
from models import build_model
from metrics import protein_hits_at_k

DATA_DIR = REPO / "data"
RUNS_DIR = REPO / "runs" / "splits_comparison_taxa" / "residual_jumping_sage_L2_h128_dot_lr0.0001"
TARGET_EDGE = ("Protein", "catalyzes", "Interaction")


def gene_product_node_ids(nodes_df: pd.DataFrame) -> list[str]:
    """GeneProduct node_ids in clean-graph order (mirrors build_clean_ctx's
    keep_mask_non_alias: drop pathway-scoped alias nodes)."""
    gp = nodes_df[nodes_df.node_type == "GeneProduct"]
    is_alias = gp["node_id"].str.contains("/Pathway/", regex=False)
    return gp.loc[~is_alias, "node_id"].tolist()


def held_out_node_sets(seed_hp: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (held_out_protein_idx, held_out_geneproduct_idx) as LongTensors,
    in the *clean*-graph index space used by ctx.val_data / ctx.test_data."""
    raw_data = torch.load(DATA_DIR / "heterodata.pt", weights_only=False)
    splits = torch.load(DATA_DIR / "splits_taxa.pt", weights_only=False)
    held_out_orgs = set(splits["val_organisms"]) | set(splits["test_organisms"])

    # Protein indices are unmodified by clean-graph filtering, so raw == clean.
    po_ei = raw_data[("Protein", "organism", "Organism")].edge_index
    held_out_proteins = {p for p, o in zip(po_ei[0].tolist(), po_ei[1].tolist()) if o in held_out_orgs}

    # GeneProduct indices ARE reindexed by clean-graph filtering; map via node_id.
    go_ei = raw_data[("GeneProduct", "organism", "Organism")].edge_index
    held_out_gp_raw = {g for g, o in zip(go_ei[0].tolist(), go_ei[1].tolist()) if o in held_out_orgs}

    nodes = pd.read_csv(DATA_DIR / "nodes.tsv", sep="\t", low_memory=False)
    gp_raw_ids = nodes.loc[nodes.node_type == "GeneProduct", "node_id"].tolist()
    held_out_gp_ids = {gp_raw_ids[i] for i in held_out_gp_raw}

    clean_gp_ids = gene_product_node_ids(nodes)
    held_out_gp_clean = {i for i, nid in enumerate(clean_gp_ids) if nid in held_out_gp_ids}

    print(f"  Held-out organisms       : {len(held_out_orgs)} "
          f"(val {len(splits['val_organisms'])} + test {len(splits['test_organisms'])})")
    print(f"  Held-out Protein nodes    : {len(held_out_proteins):,}")
    print(f"  Held-out GeneProduct nodes: {len(held_out_gp_clean):,}")

    return (torch.tensor(sorted(held_out_proteins), dtype=torch.long),
            torch.tensor(sorted(held_out_gp_clean), dtype=torch.long))


def strip_neighborhood(data, held_out_protein_idx: torch.Tensor, held_out_gp_idx: torch.Tensor):
    """Remove every edge (any type except TARGET_EDGE) touching a held-out
    Protein or GeneProduct node, as either source or destination."""
    d = data.clone()
    for et in list(d.edge_types):
        if et == TARGET_EDGE:
            continue
        src_type, _, dst_type = et
        ei = d[et].edge_index
        if ei.numel() == 0:
            continue
        keep = torch.ones(ei.shape[1], dtype=torch.bool, device=ei.device)
        if src_type == "Protein":
            keep &= ~torch.isin(ei[0], held_out_protein_idx.to(ei.device))
        if dst_type == "Protein":
            keep &= ~torch.isin(ei[1], held_out_protein_idx.to(ei.device))
        if src_type == "GeneProduct":
            keep &= ~torch.isin(ei[0], held_out_gp_idx.to(ei.device))
        if dst_type == "GeneProduct":
            keep &= ~torch.isin(ei[1], held_out_gp_idx.to(ei.device))
        d[et].edge_index = ei[:, keep]
    return d


def evaluate(model, ctx, data, k_list=(1, 5, 10, 50)):
    return protein_hits_at_k(model.gnn, model.predictor, data, ctx,
                              k_list=k_list, eval_embedded_only=True, any_catalyst_lookup=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1, 2, 3])
    ap.add_argument("--k-list", nargs="+", type=int, default=[1, 5, 10, 50])
    args = ap.parse_args()

    print("Identifying held-out organism node sets …")
    held_out_protein_idx = held_out_gp_idx = None

    results_baseline, results_isolated = [], []

    for seed in args.seeds:
        ckpt_dir = RUNS_DIR / f"seed_{seed}"
        ckpt_files = sorted(p for p in ckpt_dir.glob("*best*.pt")
                             if "superseded" not in str(p))
        assert ckpt_files, f"no checkpoint in {ckpt_dir}"
        ckpt_path = ckpt_files[-1]
        print(f"\n=== seed {seed}: {ckpt_path.name} ===")
        ckpt = torch.load(ckpt_path, weights_only=False)
        hp = ckpt.get("hparams", {})

        ctx = load_ctx_for_run(hp, DATA_DIR, seed=seed, download=False, print_summary=False)
        if held_out_protein_idx is None:
            held_out_protein_idx, held_out_gp_idx = held_out_node_sets(hp)

        model = build_model(ctx, gnn_name=hp.get("gnn_name", "sage"),
                             hidden_dim=hp.get("hidden_dim", 128), num_layers=hp.get("num_layers", 2),
                             decoder=hp.get("decoder", "dot"), dropout=hp.get("dropout", 0.3),
                             random_seed=seed, use_norm=hp.get("layer_norm", False))
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        base_val = evaluate(model, ctx, ctx.val_data, args.k_list)
        base_test = evaluate(model, ctx, ctx.test_data, args.k_list)
        print(f"  baseline : val P-H@50={base_val['P-H@50']*100:.2f}%  test P-H@50={base_test['P-H@50']*100:.2f}%")

        iso_val_data = strip_neighborhood(ctx.val_data, held_out_protein_idx, held_out_gp_idx)
        iso_test_data = strip_neighborhood(ctx.test_data, held_out_protein_idx, held_out_gp_idx)
        iso_val = evaluate(model, ctx, iso_val_data, args.k_list)
        iso_test = evaluate(model, ctx, iso_test_data, args.k_list)
        print(f"  isolated : val P-H@50={iso_val['P-H@50']*100:.2f}%  test P-H@50={iso_test['P-H@50']*100:.2f}%")

        results_baseline.append({"seed": seed, "val": base_val, "test": base_test})
        results_isolated.append({"seed": seed, "val": iso_val, "test": iso_test})

    def summarize(results, split, metric):
        vals = [r[split][metric] * 100 for r in results]
        return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    print(f"\n{'=' * 70}\nSummary across {len(args.seeds)} seeds\n{'=' * 70}")
    for k in args.k_list:
        metric = f"P-H@{k}"
        bv_m, bv_s = summarize(results_baseline, "val", metric)
        bt_m, bt_s = summarize(results_baseline, "test", metric)
        iv_m, iv_s = summarize(results_isolated, "val", metric)
        it_m, it_s = summarize(results_isolated, "test", metric)
        print(f"{metric:<8}  baseline val {bv_m:6.2f}+/-{bv_s:4.2f}  test {bt_m:6.2f}+/-{bt_s:4.2f}"
              f"   |   isolated val {iv_m:6.2f}+/-{iv_s:4.2f}  test {it_m:6.2f}+/-{it_s:4.2f}")


if __name__ == "__main__":
    main()
