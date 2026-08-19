#!/usr/bin/env python3
"""Precompute EC hierarchy one-hot features for all Interaction nodes.

Parses data/interim/reactions.ttl once with rdflib (~30-60 s), builds a
237-dim one-hot encoding per Interaction node, and saves two outputs:

  data/processed/embeddings_ec.pt
      dict {node_uri_str → 237-dim float32 tensor}  (same format as
      embeddings_conversion.pt).  Loaded automatically by fulldata_baseline_train.py
      when --ec-features is passed, skipping the TTL parse entirely.

  data/processed/nodes.tsv  (UPDATED IN-PLACE)
      Adds an `ec_number` column to the existing file.  All Interaction rows
      get the full EC annotation (e.g. "2.1.1.280"); non-Interaction rows and
      unannotated Interaction rows get an empty string.

Encoding (hierarchy_levels=3, the recommended default):
  L1 EC class       :  8 dims  (7 classes + 1 unknown)
  L2 EC sub-class   : 61 dims  (60 subclasses + 1 unknown)
  L3 sub-sub-class  :168 dims  (167 sub-subclasses + 1 unknown)
  Total             :237 dims

Coverage: ~55-60 % of Conversion nodes have an EC annotation; all others
get the [0,...,0,1] "unknown" one-hot at each level.

Usage:
    python scripts/compute_ec_embeddings.py
    python scripts/compute_ec_embeddings.py --ttl data/interim/reactions.ttl \\
        --data-dir data/processed --out-dir data/processed
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))

from fulldata_baseline_common import (  # noqa: E402
    build_ec_hierarchy_features,
    parse_ec_from_reactions_ttl,
)

HIERARCHY_LEVELS = 3


def compute(ttl_path: Path, data_dir: Path, out_dir: Path) -> None:
    # ── 1. Parse EC numbers from TTL ──────────────────────────────────────────
    ec_by_local = parse_ec_from_reactions_ttl(ttl_path)
    print(f"  {len(ec_by_local):,} Interaction local IDs with EC annotations")

    # ── 2. Load node list ─────────────────────────────────────────────────────
    print(f"\nLoading nodes from {data_dir / 'nodes.tsv'} ...")
    nodes_df = pd.read_csv(data_dir / "nodes.tsv", sep="\t")
    inter_df = nodes_df[nodes_df["node_type"] == "Interaction"].reset_index(drop=True)
    n_inter  = len(inter_df)
    n_conv   = (inter_df["interaction_subtype"] == "Conversion").sum()
    print(f"  {n_inter:,} Interaction nodes  ({n_conv:,} Conversion)")

    # ── 3. Build 237-dim one-hot features ─────────────────────────────────────
    print(f"\nBuilding {HIERARCHY_LEVELS}-level EC hierarchy features ...")
    node_ids = inter_df["node_id"].tolist()
    ec_feat, level_vocabs = build_ec_hierarchy_features(
        ec_by_local, node_ids, hierarchy_levels=HIERARCHY_LEVELS
    )
    total_dim = ec_feat.shape[1]
    print(f"  Feature dims: " +
          " + ".join(f"L{i+1}={len(v)+1}" for i, v in enumerate(level_vocabs)) +
          f" = {total_dim}")

    # Coverage stats
    n_with_ec = sum(1 for nid in node_ids if nid.rsplit("/", 1)[-1] in ec_by_local)
    conv_ids  = inter_df.loc[inter_df["interaction_subtype"] == "Conversion", "node_id"]
    n_conv_ec = sum(1 for nid in conv_ids if nid.rsplit("/", 1)[-1] in ec_by_local)
    print(f"  Coverage — all Interactions: {n_with_ec}/{n_inter} ({100*n_with_ec/n_inter:.1f}%)")
    print(f"  Coverage — Conversions:      {n_conv_ec}/{n_conv} ({100*n_conv_ec/n_conv:.1f}%)")

    _EC_NAMES = ["Oxidoreductases", "Transferases", "Hydrolases",
                 "Lyases", "Isomerases", "Ligases", "Translocases"]
    from collections import Counter
    cls_counts = Counter(
        int(ec_by_local[nid.rsplit("/", 1)[-1]].split(".")[0])
        for nid in conv_ids
        if nid.rsplit("/", 1)[-1] in ec_by_local
    )
    print("  EC class breakdown (Conversion nodes):")
    for c in sorted(cls_counts):
        print(f"    EC {c} ({_EC_NAMES[c-1]}): {cls_counts[c]:,}")

    # ── 4. Save embeddings_ec.pt ──────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings: dict[str, torch.Tensor] = {
        nid: ec_feat[i] for i, nid in enumerate(node_ids)
    }
    out_path = out_dir / "embeddings_ec.pt"
    torch.save(embeddings, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved {out_path}  ({size_mb:.1f} MB, {len(embeddings):,} nodes, {total_dim}-dim)")

    # ── 5. Add ec_number column to nodes.tsv ─────────────────────────────────
    nodes_path = data_dir / "nodes.tsv"
    print(f"\nUpdating {nodes_path} with ec_number column ...")
    nodes_df_updated = pd.read_csv(nodes_path, sep="\t")

    local_to_ec = ec_by_local  # {local_id → "2.1.1.280"}

    def _get_ec(row):
        if row["node_type"] != "Interaction":
            return ""
        local = str(row["node_id"]).rsplit("/", 1)[-1]
        return local_to_ec.get(local, "")

    nodes_df_updated["ec_number"] = nodes_df_updated.apply(_get_ec, axis=1)

    n_annotated = (nodes_df_updated["ec_number"] != "").sum()
    print(f"  Annotated {n_annotated:,} nodes with EC numbers")

    nodes_df_updated.to_csv(nodes_path, sep="\t", index=False)
    print(f"  Wrote updated {nodes_path}")
    print("\nDone.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ttl", type=str,
                    default=str(REPO_ROOT / "data" / "interim" / "reactions.ttl"),
                    help="path to reactions.ttl (default: data/interim/reactions.ttl)")
    ap.add_argument("--data-dir", type=str,
                    default=str(REPO_ROOT / "data" / "processed"),
                    help="directory containing nodes.tsv (default: data/processed)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="output directory (default: same as --data-dir)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir) if args.out_dir else data_dir
    compute(Path(args.ttl), data_dir, out_dir)


if __name__ == "__main__":
    main()
