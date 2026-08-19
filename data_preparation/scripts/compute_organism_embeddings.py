#!/usr/bin/env python3
"""Compute taxonomy-aware embeddings for all Organism nodes.

Two complementary representations are produced and saved to
data/processed/embeddings_organism.pt:

  "lineage_multihot"  (n_org × n_taxa) sparse-converted to dense:
      Binary vector per organism — 1 at every position corresponding to a
      taxon ID that appears in its NCBI lineage.  Only named ranks are kept
      (domain, kingdom, phylum, subphylum, class, subclass, order, family,
      tribe, genus); unnamed "clade" nodes are dropped.  Naturally captures
      hierarchical relatedness: two organisms sharing order share all bits
      above order too.

  "mds"  (n_org × mds_dim):
      Classical MDS (PCoA) on the pairwise taxonomic-hop distance matrix.
      Distance between A and B = depth(A) + depth(B) - 2·depth(LCA(A,B)),
      where depth = number of named-rank ancestors.  Continuous, low-dim,
      directly embeds how far apart organisms are in the tree.

Both tensors use the same row order as Organism nodes in nodes.tsv.

Usage:
    python scripts/compute_organism_embeddings.py [--mds-dim 64] [--out PATH]

Output:
    data/processed/embeddings_organism.pt
    → dict with keys "lineage_multihot", "mds", "taxid_order", "lineage_vocab"
"""
import argparse
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
NODES_TSV     = PROCESSED_DIR / "nodes.tsv"
DEFAULT_OUT   = PROCESSED_DIR / "embeddings_organism.pt"

NCBI_EFETCH   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_API_KEY  = os.environ.get("NCBI_API_KEY")
BATCH_SIZE    = 50   # NCBI allows up to 500 per POST; 50 is safe for XML size
REQUEST_DELAY = 0.34 if not NCBI_API_KEY else 0.11   # 3 or 10 req/s

# Ranks to keep for the multi-hot encoding (unnamed "clade" nodes excluded)
NAMED_RANKS = {
    "superkingdom", "domain", "kingdom", "subkingdom",
    "phylum", "subphylum",
    "class", "subclass", "infraclass",
    "order", "suborder",
    "family", "subfamily", "tribe", "subtribe",
    "genus", "subgenus",
    "species group", "species subgroup", "species",
}


def fetch_lineage_batch(taxids: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Fetch lineage for a batch of NCBI taxon IDs.

    Returns {taxid: [(rank, taxid_ancestor), ...]} with named ranks only,
    in order from root to leaf (excluding the organism itself).
    """
    params = {
        "db": "taxonomy",
        "id": ",".join(taxids),
        "retmode": "xml",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    for attempt in range(4):
        try:
            r = requests.get(NCBI_EFETCH, params=params, timeout=60)
            r.raise_for_status()
            break
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"    retry {attempt+1}/3 after error: {exc}")
            time.sleep(2 ** attempt)

    root = ET.fromstring(r.text)
    result = {}
    for taxon in root.findall("Taxon"):
        tid  = taxon.findtext("TaxId", "").strip()
        ancs = []
        for anc in taxon.findall(".//LineageEx/Taxon"):
            rank = anc.findtext("Rank", "").strip().lower()
            aid  = anc.findtext("TaxId", "").strip()
            if rank in NAMED_RANKS and aid:
                ancs.append((rank, aid))
        result[tid] = ancs
    return result


def build_lineage_multihot(
    lineages: dict[str, list[tuple[str, str]]],
    taxid_order: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build binary multi-hot matrix (n_org × n_vocab).

    vocab = sorted union of all ancestor taxon IDs across all organisms.
    """
    vocab_set: set[str] = set()
    for ancs in lineages.values():
        for _, aid in ancs:
            vocab_set.add(aid)
    vocab = sorted(vocab_set, key=lambda x: int(x))
    v2i   = {v: i for i, v in enumerate(vocab)}

    mat = np.zeros((len(taxid_order), len(vocab)), dtype=np.float32)
    for row, tid in enumerate(taxid_order):
        for _, aid in lineages.get(tid, []):
            if aid in v2i:
                mat[row, v2i[aid]] = 1.0
    return mat, vocab


def pairwise_hop_distance(
    lineages: dict[str, list[tuple[str, str]]],
    taxid_order: list[str],
) -> np.ndarray:
    """Symmetric pairwise distance matrix based on named-rank lineage hops.

    dist(A, B) = depth(A) + depth(B) - 2 * depth(LCA(A,B))
    where depth = number of named-rank ancestors.
    LCA is found by walking from root; the last shared ancestor taxon ID is the LCA.
    """
    n = len(taxid_order)
    # ancestor sets as ordered lists (root first)
    anc_lists = {tid: [aid for _, aid in lineages.get(tid, [])] for tid in taxid_order}

    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        ai = set(anc_lists[taxid_order[i]])
        di = len(anc_lists[taxid_order[i]])
        for j in range(i + 1, n):
            aj = set(anc_lists[taxid_order[j]])
            dj = len(anc_lists[taxid_order[j]])
            # LCA depth = number of shared ancestors (they share a contiguous prefix)
            shared = 0
            for a, b in zip(anc_lists[taxid_order[i]], anc_lists[taxid_order[j]]):
                if a == b:
                    shared += 1
                else:
                    break
            d = di + dj - 2 * shared
            dist[i, j] = dist[j, i] = d
    return dist


def classical_mds(dist: np.ndarray, dim: int) -> np.ndarray:
    """Classical MDS (PCoA) from a squared distance matrix."""
    n = dist.shape[0]
    D2 = dist ** 2
    H  = np.eye(n) - np.ones((n, n)) / n
    B  = -0.5 * H @ D2 @ H
    # Symmetric → use eigh for stability
    vals, vecs = np.linalg.eigh(B)
    # Sort descending
    idx  = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    # Keep positive eigenvalues only (up to dim)
    pos  = vals > 1e-8
    take = min(dim, int(pos.sum()))
    emb  = vecs[:, :take] * np.sqrt(np.maximum(vals[:take], 0))
    # Pad to requested dim if fewer positive eigenvalues
    if take < dim:
        pad = np.zeros((n, dim - take), dtype=np.float32)
        emb = np.concatenate([emb, pad], axis=1)
    return emb.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mds-dim", type=int, default=64,
                    help="Dimensionality of the MDS embedding (default 64)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    import re
    print("Loading nodes.tsv …")
    nodes = pd.read_csv(NODES_TSV, sep="\t", low_memory=False)
    org_nodes = nodes[nodes["node_type"] == "Organism"]["node_id"].tolist()
    taxid_order = []
    for nid in org_nodes:
        m = re.search(r"NCBITaxon_(\d+)", nid)
        if m:
            taxid_order.append(m.group(1))
        else:
            print(f"  WARNING: cannot parse taxon ID from {nid}, skipping")
    print(f"  {len(taxid_order)} organism nodes with NCBI taxon IDs")

    # ── Fetch lineages in batches ─────────────────────────────────────────────
    print(f"\nFetching lineages from NCBI (batches of {BATCH_SIZE}) …")
    lineages: dict[str, list[tuple[str, str]]] = {}
    batches = [taxid_order[i:i+BATCH_SIZE] for i in range(0, len(taxid_order), BATCH_SIZE)]
    for bi, batch in enumerate(batches):
        print(f"  batch {bi+1}/{len(batches)}  ({len(batch)} taxids) …", end=" ", flush=True)
        result = fetch_lineage_batch(batch)
        lineages.update(result)
        print(f"ok ({len(result)} returned)")
        time.sleep(REQUEST_DELAY)

    n_resolved = sum(1 for t in taxid_order if t in lineages and lineages[t])
    print(f"  Resolved: {n_resolved}/{len(taxid_order)}")

    # ── Multi-hot lineage encoding ────────────────────────────────────────────
    print("\nBuilding multi-hot lineage encoding …")
    multihot, vocab = build_lineage_multihot(lineages, taxid_order)
    print(f"  Shape: {multihot.shape}  (organisms × lineage vocab)")
    print(f"  Vocab size: {len(vocab)} unique named-rank ancestor taxon IDs")

    # ── Pairwise distance matrix ──────────────────────────────────────────────
    print("\nComputing pairwise taxonomic-hop distance matrix …")
    dist = pairwise_hop_distance(lineages, taxid_order)
    print(f"  Distance matrix: {dist.shape}")
    print(f"  Mean dist: {dist.mean():.2f}  Max: {dist.max():.0f}  "
          f"(0 = same lineage, higher = more distant)")

    # ── MDS embedding ─────────────────────────────────────────────────────────
    print(f"\nApplying classical MDS (dim={args.mds_dim}) …")
    mds_emb = classical_mds(dist, args.mds_dim)
    print(f"  MDS embedding shape: {mds_emb.shape}")

    # ── Save ──────────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lineage_multihot": torch.from_numpy(multihot),
        "mds":              torch.from_numpy(mds_emb),
        "distance_matrix":  torch.from_numpy(dist),
        "taxid_order":      taxid_order,    # list[str] — same row order as nodes.tsv Organism rows
        "lineage_vocab":    vocab,           # list[str] — taxon IDs for multihot columns
    }
    torch.save(payload, args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nSaved → {args.out}  ({size_mb:.1f} MB)")
    print("\nKeys in saved dict:")
    for k, v in payload.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:<20} {tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:<20} list[{type(v[0]).__name__}] len={len(v)}")


if __name__ == "__main__":
    main()
