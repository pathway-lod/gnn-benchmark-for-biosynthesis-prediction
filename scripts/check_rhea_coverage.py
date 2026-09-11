#!/usr/bin/env python3
"""Quick coverage check for the Rhea validation strategy.

Downloads Rhea TSV files, parses TTL xrefs, and reports how many
ath_pathway test reactions have Rhea annotations — without loading a model.

Usage:
    python scripts/check_rhea_coverage.py [--rhea-dir data/rhea_cache]
"""
from __future__ import annotations
import re
import sys
import urllib.request
from pathlib import Path
from collections import defaultdict

import torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "training"))
from dataset import load_data
from utils import set_seed

RHEA_FTP = "https://ftp.expasy.org/databases/rhea/tsv"
RHEA_METACYC_URL = f"{RHEA_FTP}/rhea2metacyc.tsv"
RHEA_UNIPROT_URL = f"{RHEA_FTP}/rhea2uniprot_sprot.tsv"


def download_if_missing(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  → {dest}")
    else:
        print(f"  Cached: {dest.name}")
    return dest


def load_rhea_metacyc(tsv: Path) -> dict[str, set[str]]:
    m: dict[str, set[str]] = defaultdict(set)
    with open(tsv) as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            rhea_id, direction, master_id, metacyc_id = parts[:4]
            if metacyc_id and direction == "UN":
                m[metacyc_id].add(master_id)
    return dict(m)


def load_rhea_uniprot(tsv: Path) -> dict[str, set[str]]:
    m: dict[str, set[str]] = defaultdict(set)
    with open(tsv) as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            _, direction, master_id, acc = parts[:4]
            if master_id and acc and direction == "UN":
                m[master_id].add(acc)
    return dict(m)


def parse_uniprot_from_ttl(ttl: Path) -> dict[str, list[str]]:
    content = ttl.read_text()
    result: dict[str, list[str]] = {}
    for block in re.split(r"\n(?=<)", content):
        m = re.match(r"<([^>]+)>", block)
        if not m or "AlternativeId_Uniprot" not in block:
            continue
        local = m.group(1).split("/")[-1]
        accs = re.findall(
            r'pmw:key\s+"AlternativeId_Uniprot"\s*;\s*pmw:value\s+"([^"]+)"',
            block,
        )
        if not accs:
            accs = re.findall(
                r'pmw:value\s+"([A-Z][A-Z0-9]{5,9})"\s*;\s*pmw:key\s+"AlternativeId_Uniprot"',
                block,
            )
        if accs:
            result[local] = accs
    return result


def parse_rhea_from_ttl(ttl: Path) -> dict[str, set[str]]:
    content = ttl.read_text()
    m: dict[str, set[str]] = defaultdict(set)
    for block in re.split(r"\n(?=<)", content):
        bm = re.match(r"<([^>]+)>", block)
        if not bm or '"Rhea"' not in block:
            continue
        local = bm.group(1).split("/")[-1]
        rhea_ids = re.findall(r'xrefId\s+"(\d+)"', block)
        if not rhea_ids:
            continue
        parent = re.sub(r"_anchor_(reactant|product)_\d+$", "", local)
        parent = re.sub(r"_ENZRXN_\d+_\d+_\d+$", "", parent)
        m[parent].update(rhea_ids)
    return dict(m)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhea-dir", default=str(REPO / "data" / "rhea_cache"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rhea_dir = Path(args.rhea_dir)
    rhea_dir.mkdir(parents=True, exist_ok=True)

    # ── Download Rhea ──────────────────────────────────────────────────────────
    print("── Downloading Rhea files ────────────────────────────────────────────────")
    mc_tsv  = download_if_missing(RHEA_METACYC_URL, rhea_dir / "rhea2metacyc.tsv")
    up_tsv  = download_if_missing(RHEA_UNIPROT_URL, rhea_dir / "rhea2uniprot_sprot.tsv")

    metacyc_to_rhea  = load_rhea_metacyc(mc_tsv)
    rhea_to_uniprot  = load_rhea_uniprot(up_tsv)
    print(f"  Rhea-MetaCyc entries : {len(metacyc_to_rhea):,}")
    print(f"  Rhea-UniProt entries : {len(rhea_to_uniprot):,}")

    # ── Parse TTL xrefs ────────────────────────────────────────────────────────
    print()
    print("── Parsing TTL xrefs ─────────────────────────────────────────────────────")
    prop_ttl = REPO.parent / "data" / "interim" / "properties_extra.ttl"
    rxn_ttl  = REPO.parent / "data" / "interim" / "reactions.ttl"

    local_to_uniprot: dict[str, list[str]] = {}
    if prop_ttl.exists():
        print(f"  Parsing {prop_ttl.name} …")
        local_to_uniprot = parse_uniprot_from_ttl(prop_ttl)
        print(f"    {len(local_to_uniprot):,} protein nodes with UniProt")
    else:
        print(f"  WARNING: {prop_ttl} not found")

    ttl_rxn_to_rhea: dict[str, set[str]] = {}
    if rxn_ttl.exists():
        print(f"  Parsing {rxn_ttl.name} for Rhea xrefs …")
        ttl_rxn_to_rhea = parse_rhea_from_ttl(rxn_ttl)
        print(f"    {len(ttl_rxn_to_rhea):,} reactions with Rhea xrefs in TTL")

    # ── Load ath_pathway split (to know test reaction node indices) ────────────
    print()
    print("── Loading ath_pathway split ─────────────────────────────────────────────")
    set_seed(args.seed)
    ctx = load_data(
        data_dir=str(REPO / "data"),
        random_seed=args.seed,
        split_type="ath_pathway",
        species_pool=True,
        remove_is_part_of=True,
        embedded_only_ranking=True,
        download=False,
    )
    test_rxn_idxs = ctx.test_data[ctx.target_edge].edge_label_index[1][
        ctx.test_data[ctx.target_edge].edge_label == 1
    ].unique().tolist()
    print(f"  Test reactions (unique): {len(test_rxn_idxs)}")

    # ── Load nodes.tsv for local-id mapping ───────────────────────────────────
    import pandas as pd
    nodes_df = pd.read_csv(REPO / "data" / "nodes.tsv", sep="\t", low_memory=False)
    prot_rows  = nodes_df[nodes_df.node_type == "Protein"].reset_index(drop=True)
    inter_rows = nodes_df[nodes_df.node_type == "Interaction"].reset_index(drop=True)
    inter_local = [iri.split("/")[-1] for iri in inter_rows.node_id]
    prot_local  = [iri.split("/")[-1] for iri in prot_rows.node_id]

    uniprot_to_idx: dict[str, set[int]] = defaultdict(set)
    for local_id, accs in local_to_uniprot.items():
        idx = next((i for i, l in enumerate(prot_local) if l == local_id), None)
        if idx is None:
            continue
        for acc in accs:
            uniprot_to_idx[acc].add(idx)
    print(f"  Proteins with UniProt mapping: "
          f"{len({i for s in uniprot_to_idx.values() for i in s}):,} nodes, "
          f"{len(uniprot_to_idx):,} accessions")

    # ── Coverage over test reactions ───────────────────────────────────────────
    print()
    print("── Coverage of test reactions ────────────────────────────────────────────")
    n_ttl = n_mc = n_either = n_with_protein = 0
    for rxn_idx in test_rxn_idxs:
        local = inter_local[rxn_idx] if rxn_idx < len(inter_local) else ""
        rhea_form = local.replace("_", "-")
        ttl_ids = ttl_rxn_to_rhea.get(local, set())
        mc_ids  = metacyc_to_rhea.get(local, set()) | metacyc_to_rhea.get(rhea_form, set())
        all_ids = ttl_ids | mc_ids
        if ttl_ids:
            n_ttl += 1
        if mc_ids:
            n_mc += 1
        if all_ids:
            n_either += 1
            uniprots = set()
            for rid in all_ids:
                uniprots |= rhea_to_uniprot.get(rid, set())
            prot_idxs = set()
            for acc in uniprots:
                prot_idxs |= uniprot_to_idx.get(acc, set())
            if prot_idxs:
                n_with_protein += 1

    n = len(test_rxn_idxs)
    print(f"  Reactions with Rhea ID (TTL xrefs only) : {n_ttl}/{n} ({n_ttl/n*100:.1f}%)")
    print(f"  Reactions with Rhea ID (Rhea-MetaCyc)   : {n_mc}/{n} ({n_mc/n*100:.1f}%)")
    print(f"  Reactions with Rhea ID (either source)  : {n_either}/{n} ({n_either/n*100:.1f}%)")
    print(f"  Reactions with ≥1 pool protein via Rhea : {n_with_protein}/{n} ({n_with_protein/n*100:.1f}%)")
    print()
    print("If coverage is <10%, the Rhea validation will cover too few test reactions")
    print("to be meaningful as a standalone metric, but can still report as a subset.")


if __name__ == "__main__":
    main()
