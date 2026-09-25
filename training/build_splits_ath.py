#!/usr/bin/env python3
"""Build a within-species pathway holdout split for Arabidopsis thaliana.

Uses only catalysis edges where the protein belongs to A. thaliana
(NCBITaxon_3702).  Reactions (Interaction nodes) are the unit of
holdout: a reaction is assigned to exactly one split, and all edges
involving that reaction land in the same split.  Stratification is by
top-level EC class (EC1 … EC7 + unknown) so that each split sees the
same EC distribution.

Output
------
data/splits_ath_pathway.pt
  {
    "target_edge": ("Protein", "catalyzes", "Interaction"),
    "train_edge_index": LongTensor [2, n_train],
    "val_edge_index":   LongTensor [2, n_val],
    "test_edge_index":  LongTensor [2, n_test],
    "ath_protein_idxs": LongTensor [n_ath_proteins_in_pool],
  }

Usage
-----
    python training/build_splits_ath.py
    python training/build_splits_ath.py --data-dir /path/to/data --seed 42
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TARGET_EDGE  = ("Protein", "catalyzes", "Interaction")
ATH_IRI      = "http://purl.obolibrary.org/obo/NCBITaxon_3702"
DATA_DIR_DEF = Path(__file__).parent.parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR_DEF, type=Path)
    parser.add_argument("--seed",     default=42, type=int,
                        help="Random seed for reaction shuffle (default 42)")
    parser.add_argument("--val-frac",  default=0.10, type=float)
    parser.add_argument("--test-frac", default=0.20, type=float)
    parser.add_argument("--output",   default=None, type=Path,
                        help="Output path (default: data/splits_ath_pathway.pt)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path  = args.output or data_dir / "splits_ath_pathway.pt"

    # ── Load graph ────────────────────────────────────────────────────────
    print("Loading heterodata.pt …")
    data = torch.load(data_dir / "heterodata.pt", weights_only=False)
    print("Loading nodes.tsv …")
    nodes_df = pd.read_csv(data_dir / "nodes.tsv", sep="\t", low_memory=False)
    print("Loading embeddings_protein.pt …")
    prot_emb: dict = torch.load(data_dir / "embeddings_protein.pt", weights_only=False)

    # ── Protein index maps ────────────────────────────────────────────────
    prot_ids  = nodes_df.loc[nodes_df["node_type"] == "Protein", "node_id"].tolist()
    iri2pidx  = {iri: i for i, iri in enumerate(prot_ids)}

    # ── Ranking pool: proteins with non-zero ESM-C norm ───────────────────
    emb_dim = next(iter(prot_emb.values())).shape[0]
    feat = torch.zeros(len(prot_ids), emb_dim)
    for i, nid in enumerate(prot_ids):
        if nid in prot_emb:
            feat[i] = prot_emb[nid]
    pool_mask = feat.norm(dim=-1) > 0   # [n_prot] bool, same as _build_embedded_mask_from_x

    # ── Identify A. thaliana proteins ─────────────────────────────────────
    org_ids    = nodes_df.loc[nodes_df["node_type"] == "Organism", "node_id"].tolist()
    ath_org_idx = org_ids.index(ATH_IRI)
    prot_org_ei = data[("Protein", "organism", "Organism")].edge_index
    ath_prot_idx_set = set(
        prot_org_ei[0][prot_org_ei[1] == ath_org_idx].tolist()
    )
    print(f"  A. thaliana proteins      : {len(ath_prot_idx_set):,}")

    # Intersection with pool (non-zero ESM-C)
    ath_pool_idxs = torch.tensor(
        sorted(i for i in ath_prot_idx_set if pool_mask[i]),
        dtype=torch.long,
    )
    print(f"  A. thaliana in ESM-C pool : {len(ath_pool_idxs):,}  "
          f"(random P-H@50={50/len(ath_pool_idxs):.2%})")

    # ── Conversion-subtype reaction index set (matches benchmark evaluation) ─
    inter_ids = nodes_df.loc[nodes_df["node_type"] == "Interaction", "node_id"].tolist()
    inter_ec  = nodes_df.loc[nodes_df["node_type"] == "Interaction", "ec_number"].tolist()
    inter_sub = nodes_df.loc[nodes_df["node_type"] == "Interaction", "interaction_subtype"].tolist()
    conv_idx_set = {
        i for i, sub in enumerate(inter_sub) if sub == "Conversion"
    }
    print(f"  Conversion-subtype reactions in graph : {len(conv_idx_set):,}")

    # ── Get A. thaliana catalysis edges: ESM-C protein AND Conversion target ─
    cat_ei = data[TARGET_EDGE].edge_index   # [protein_idx, interaction_idx]
    ath_esmc_conv_mask = torch.tensor([
        int(p) in ath_prot_idx_set and pool_mask[int(p)] and int(r) in conv_idx_set
        for p, r in zip(cat_ei[0].tolist(), cat_ei[1].tolist())
    ])
    ath_ei = cat_ei[:, ath_esmc_conv_mask]   # [2, n_ath_edges]
    n_edges = ath_ei.shape[1]
    print(f"  A. thaliana catalysis edges (ESM-C + Conversion) : {n_edges:,}")

    # ── Build reaction → edges map ────────────────────────────────────────
    # Map interaction index → EC top-level class (1-7, or 0 for unknown)
    inter_idx2ec = {}
    for i, (nid, ec) in enumerate(zip(inter_ids, inter_ec)):
        try:
            top = int(str(ec).split(".")[0])
        except (ValueError, AttributeError):
            top = 0
        inter_idx2ec[i] = top

    rxn_to_edges: dict[int, list[int]] = collections.defaultdict(list)
    for edge_i, (_, rxn_idx) in enumerate(zip(ath_ei[0].tolist(), ath_ei[1].tolist())):
        rxn_to_edges[int(rxn_idx)].append(edge_i)
    rxns = sorted(rxn_to_edges.keys())
    print(f"  Unique A. thaliana reactions : {len(rxns):,}")

    # ── Stratify reactions by EC class, then split ────────────────────────
    rng = np.random.default_rng(args.seed)
    # Group reactions by EC top-level class
    ec_to_rxns: dict[int, list[int]] = collections.defaultdict(list)
    for rxn in rxns:
        ec_to_rxns[inter_idx2ec[rxn]].append(rxn)

    train_rxns, val_rxns, test_rxns = [], [], []
    for ec_class in sorted(ec_to_rxns.keys()):
        group = np.array(ec_to_rxns[ec_class])
        rng.shuffle(group)
        n = len(group)
        n_val  = max(1, round(n * args.val_frac))
        n_test = max(1, round(n * args.test_frac))
        n_val  = min(n_val,  n - 2)   # keep at least 1 train + 1 test
        n_test = min(n_test, n - 1 - n_val)
        test_rxns  += group[:n_test].tolist()
        val_rxns   += group[n_test:n_test + n_val].tolist()
        train_rxns += group[n_test + n_val:].tolist()

    def _collect_edges(rxn_list: list[int]) -> torch.Tensor:
        edge_idxs = []
        for r in rxn_list:
            edge_idxs.extend(rxn_to_edges[r])
        if not edge_idxs:
            return torch.zeros(2, 0, dtype=torch.long)
        return ath_ei[:, torch.tensor(sorted(edge_idxs), dtype=torch.long)]

    train_ei = _collect_edges(train_rxns)
    val_ei   = _collect_edges(val_rxns)
    test_ei  = _collect_edges(test_rxns)

    print()
    print("─" * 60)
    print("Split summary")
    print("─" * 60)
    total_rxns = len(rxns)
    for name, ei, rxn_list in [("train", train_ei, train_rxns),
                                ("val",   val_ei,   val_rxns),
                                ("test",  test_ei,  test_rxns)]:
        print(f"  {name:6s}  reactions={len(rxn_list):,}  ({len(rxn_list)/total_rxns:.1%})  "
              f"edges={ei.shape[1]:,}  ({ei.shape[1]/n_edges:.1%})")
    print("─" * 60)
    print(f"  Total   reactions={total_rxns:,}  edges={n_edges:,}")
    print()

    # ── EC distribution check ─────────────────────────────────────────────
    print("EC class distribution per split:")
    for split_name, rxn_list in [("train", train_rxns),
                                  ("val",   val_rxns),
                                  ("test",  test_rxns)]:
        ec_counts = collections.Counter(inter_idx2ec[r] for r in rxn_list)
        dist = "  ".join(f"EC{k}={v}" for k, v in sorted(ec_counts.items()) if k > 0)
        print(f"  {split_name:6s}: {dist}  unknown={ec_counts.get(0, 0)}")

    # ── Save ──────────────────────────────────────────────────────────────
    payload = {
        "target_edge":        TARGET_EDGE,
        "train_edge_index":   train_ei,
        "val_edge_index":     val_ei,
        "test_edge_index":    test_ei,
        "ath_protein_idxs":   ath_pool_idxs,  # for restricting ranking pool
        "ath_org_idx":        ath_org_idx,
        "seed":               args.seed,
        "val_frac":           args.val_frac,
        "test_frac":          args.test_frac,
    }
    torch.save(payload, out_path)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
