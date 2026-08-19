#!/usr/bin/env python3
"""Precompute substrate-product reaction fingerprints for all Conversion nodes.

For each Conversion (biochemical reaction), aggregates the MAP4 1024-dim embeddings
of its substrate (source) and product (target) metabolites into a 3072-dim fingerprint:

    [mean(substrate_MAP4), mean(product_MAP4), mean(substrate_MAP4) − mean(product_MAP4)]

This approximates the DRFP (Differential Reaction FingerPrint) concept — capturing the
asymmetric chemical transformation that defines a reaction — using MAP4 rather than
Morgan fingerprints as the molecular representation. The symmetric difference of atom
environments between substrates and products is implicit in the [src, tgt, src−tgt] concat.

Fallback logic (in order):
  1. Both source AND target metabolites have MAP4 → full 3072-dim fingerprint
  2. Only target has MAP4 → [zeros, target_mean, −target_mean]
  3. Only participants (no directional edges) have MAP4 → [part_mean, zeros, zeros]
  4. No MAP4 coverage → all-zeros 3072-dim vector (flagged in coverage report)

Coverage:
  ~90 % of the 19,927 Conversion nodes have ≥1 MAP4-embedded metabolite participant.
  The remaining ~10 % are reactions involving only PlantCyc-local compound IDs that
  could not be resolved to a MetaNetX/MAP4 representation.

Usage:
    python scripts/compute_conversion_embeddings.py
    python scripts/compute_conversion_embeddings.py --data-dir data/processed --out-dir data/processed

Output:
    data/processed/embeddings_conversion.pt   (dict {node_uri → 3072-dim float32 tensor})

This file is included in the Zenodo dataset so learnathon participants can use
precomputed reaction features without needing the MAP4 pipeline.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch


DIM = 1024   # MAP4 embedding dimension
OUT_DIM = 3 * DIM   # substrate | product | diff


def compute(data_dir: Path, out_dir: Path) -> None:
    print(f"Loading nodes and edges from {data_dir} ...")
    nodes_df = pd.read_csv(data_dir / "nodes.tsv", sep="\t")
    edges_df = pd.read_csv(data_dir / "edges.tsv", sep="\t")

    conv_ids = set(nodes_df.loc[nodes_df["interaction_subtype"] == "Conversion", "node_id"])
    print(f"  Total Conversion nodes: {len(conv_ids):,}")

    met_emb_path = data_dir / "embeddings_metabolite.pt"
    print(f"Loading metabolite MAP4 embeddings from {met_emb_path} ...")
    met_emb_raw: dict[str, torch.Tensor] = torch.load(met_emb_path, weights_only=False)
    # MAP4 fingerprints are stored as integer count vectors — cast to float32 for arithmetic
    met_emb = {k: v.float() for k, v in met_emb_raw.items()}
    print(f"  {len(met_emb):,} metabolites with MAP4 embeddings (dtype: {next(iter(met_emb.values())).dtype})")

    # Collect directed metabolite edges for Conversion nodes
    # source edges: (Interaction)→(Metabolite) with rel='source'
    # target edges: (Interaction)→(Metabolite) with rel='target'
    # participants: fallback, undirected participation
    rxn_edges = edges_df[
        (edges_df["src_type"] == "Interaction") &
        (edges_df["rel"].isin(["source", "target", "participants"])) &
        (edges_df["src"].isin(conv_ids))
    ].copy()

    src_edges  = rxn_edges[rxn_edges["rel"] == "source"]
    tgt_edges  = rxn_edges[rxn_edges["rel"] == "target"]
    part_edges = rxn_edges[rxn_edges["rel"] == "participants"]

    def _build_map(edge_df: pd.DataFrame) -> dict[str, list[torch.Tensor]]:
        out: dict[str, list[torch.Tensor]] = {}
        for conv_id, met_id in zip(edge_df["src"], edge_df["dst"]):
            if met_id in met_emb:
                out.setdefault(conv_id, []).append(met_emb[met_id])
        return out

    src_map  = _build_map(src_edges)
    tgt_map  = _build_map(tgt_edges)
    part_map = _build_map(part_edges)

    embeddings: dict[str, torch.Tensor] = {}
    n_full = n_partial = n_fallback = n_zero = 0

    for conv_id in sorted(conv_ids):
        srcs = src_map.get(conv_id, [])
        tgts = tgt_map.get(conv_id, [])
        parts = part_map.get(conv_id, [])

        src_emb = torch.stack(srcs).mean(dim=0) if srcs else torch.zeros(DIM)
        tgt_emb = torch.stack(tgts).mean(dim=0) if tgts else torch.zeros(DIM)

        if srcs and tgts:
            n_full += 1
        elif srcs or tgts:
            n_partial += 1
        elif parts:
            # No source/target directional edges — use participant mean as substrate proxy
            src_emb = torch.stack(parts).mean(dim=0)
            n_fallback += 1
        else:
            n_zero += 1

        diff = src_emb - tgt_emb
        embeddings[conv_id] = torch.cat([src_emb, tgt_emb, diff]).float()

    out_path = out_dir / "embeddings_conversion.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, out_path)

    total = len(conv_ids)
    print(f"\nConversion embedding coverage ({total:,} total):")
    print(f"  Full source+target MAP4 : {n_full:,}  ({n_full/total:.1%})")
    print(f"  Partial (src OR tgt)    : {n_partial:,}  ({n_partial/total:.1%})")
    print(f"  Participants-only fallback: {n_fallback:,}  ({n_fallback/total:.1%})")
    print(f"  Zero (no MAP4 coverage) : {n_zero:,}  ({n_zero/total:.1%})")
    print(f"\nOutput ({OUT_DIM}-dim = 3×MAP4): {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/processed",
                    help="directory containing nodes.tsv, edges.tsv, embeddings_metabolite.pt")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: same as --data-dir)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir
    compute(data_dir, out_dir)


if __name__ == "__main__":
    main()
