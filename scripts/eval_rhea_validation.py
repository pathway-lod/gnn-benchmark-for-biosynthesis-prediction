#!/usr/bin/env python3
"""Rhea-based independent validation of model predictions.

For each A. thaliana test reaction, checks whether the model's top-50 ranked
proteins are supported by Rhea — an external, independently curated enzyme
database.  Reactions the model ranks correctly but that are absent from
PlantMetWiki count as additional true positives, giving an upper-bound estimate
of true model quality.

Strategy
--------
1. Load Rhea-MetaCyc cross-reference (rhea-metacyc.tsv) from Rhea FTP.
2. Load rhea2uniprot.tsv to map Rhea reactions → UniProt enzyme accessions.
3. Parse properties_extra.ttl for local protein node-id → UniProt accession.
4. Match test reaction MetaCyc IDs → Rhea IDs → UniProt accessions → pool protein indices.
5. For each test reaction rank the full pool, then compute:
   - Standard   P-H@50  (PlantMetWiki labels only)
   - anyP-H@50  (all PlantMetWiki-known catalysts)
   - Rhea-val   P-H@50  (Rhea-annotated proteins, independent validation)

Usage
-----
    python scripts/eval_rhea_validation.py \
        --run-dir runs/ath_sage/sage_L2_h128_dot_lr0.0001/seed_42 \
        --data-dir data \
        [--rhea-dir /path/to/cached/rhea/tsv]   # skip download if already cached

The Rhea TSV files are downloaded once to --rhea-dir (default: data/rhea_cache/)
and reused on subsequent calls.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "training"))

from build_clean_ctx import interaction_node_ids, load_ctx_for_run
from metrics import protein_hits_at_k
from models import build_model
from utils import set_seed

# ── Rhea FTP URLs ──────────────────────────────────────────────────────────────
RHEA_FTP = "https://ftp.expasy.org/databases/rhea/tsv"
RHEA_METACYC_URL  = f"{RHEA_FTP}/rhea2metacyc.tsv"
RHEA_UNIPROT_URL  = f"{RHEA_FTP}/rhea2uniprot_sprot.tsv"   # SwissProt only (smaller)
RHEA_UNIPROT_ALL_URL = f"{RHEA_FTP}/rhea2uniprot.tsv"      # includes TrEMBL


# ── Helpers ────────────────────────────────────────────────────────────────────

def download_if_missing(url: str, dest: Path, force: bool = False) -> Path:
    if not dest.exists() or force:
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved → {dest}")
    else:
        print(f"  Using cached {dest.name}")
    return dest


def load_rhea_metacyc(tsv_path: Path) -> dict[str, set[str]]:
    """Return MetaCyc_reaction_ID -> set of Rhea master-reaction IDs."""
    # Columns: RHEA_ID  DIRECTION  MASTER_ID  MetaCyc_ID
    # DIRECTION values: UN, LR, RL, BI
    mapping: dict[str, set[str]] = defaultdict(set)
    with open(tsv_path) as f:
        header = f.readline()   # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            rhea_id, direction, master_id, metacyc_id = parts[:4]
            if metacyc_id and direction == "UN":  # master (undirected) entries only
                mapping[metacyc_id].add(master_id)
    return dict(mapping)


def load_rhea_uniprot(tsv_path: Path) -> dict[str, set[str]]:
    """Return Rhea_master_ID -> set of UniProt accessions."""
    # Columns: RHEA_ID  DIRECTION  MASTER_ID  ID  (UniProt acc)
    mapping: dict[str, set[str]] = defaultdict(set)
    with open(tsv_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            rhea_id, direction, master_id, uniprot_acc = parts[:4]
            if master_id and uniprot_acc and direction == "UN":
                mapping[master_id].add(uniprot_acc)
    return dict(mapping)


def parse_uniprot_from_ttl(ttl_path: Path) -> dict[str, list[str]]:
    """Return local_node_id -> [UniProt accessions] from properties_extra.ttl."""
    with open(ttl_path) as f:
        content = f.read()

    blocks = re.split(r"\n(?=<)", content)
    result: dict[str, list[str]] = {}

    for block in blocks:
        m = re.match(r"<([^>]+)>", block)
        if not m or "AlternativeId_Uniprot" not in block:
            continue
        iri = m.group(1)
        local = iri.split("/")[-1]

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


def parse_rhea_from_ttl(ttl_path: Path) -> dict[str, set[str]]:
    """Return local_reaction_id -> set of Rhea IDs from the Interaction TTL."""
    with open(ttl_path) as f:
        content = f.read()

    blocks = re.split(r"\n(?=<)", content)
    rxn_to_rhea: dict[str, set[str]] = defaultdict(set)

    for block in blocks:
        m = re.match(r"<([^>]+)>", block)
        if not m or '"Rhea"' not in block:
            continue
        local = m.group(1).split("/")[-1]
        rhea_ids = re.findall(r'xrefId\s+"(\d+)"', block)
        if not rhea_ids:
            continue
        # Strip anchor suffixes → parent reaction ID
        parent = re.sub(r"_anchor_(reactant|product)_\d+$", "", local)
        parent = re.sub(r"_ENZRXN_\d+_\d+_\d+$", "", parent)
        rxn_to_rhea[parent].update(rhea_ids)

    return dict(rxn_to_rhea)


# ── Main ───────────────────────────────────────────────────────────────────────

def get_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="Path to a completed seed run directory (contains report.json + *.pt checkpoint)")
    ap.add_argument("--data-dir", default=str(REPO / "data"),
                    help="PlantMetBench data directory (default: data/)")
    ap.add_argument("--rhea-dir", default=None,
                    help="Directory for cached Rhea TSV files (default: data/rhea_cache/)")
    ap.add_argument("--use-trembl", action="store_true", default=False,
                    help="Use full rhea2uniprot.tsv (includes TrEMBL); default: SwissProt only")
    ap.add_argument("--k-list", default="1,5,10,50")
    ap.add_argument("--split", choices=["val", "test", "both"], default="test")
    return ap.parse_args()


def main():
    args = get_args()
    k_list = tuple(int(k) for k in args.k_list.split(","))
    run_dir  = Path(args.run_dir)
    data_dir = Path(args.data_dir)
    rhea_dir = Path(args.rhea_dir) if args.rhea_dir else data_dir / "rhea_cache"
    rhea_dir.mkdir(parents=True, exist_ok=True)

    # ── Load run config ────────────────────────────────────────────────────────
    report_files = sorted(run_dir.glob("report*.json"))
    assert report_files, f"No report*.json in {run_dir}"
    hp = json.loads(report_files[-1].read_text())["hparams"]
    seed = hp["seed"]

    print(f"Run      : {run_dir}")
    print(f"Seed     : {seed}")
    print()

    # ── Download Rhea files ────────────────────────────────────────────────────
    print("── Rhea data ─────────────────────────────────────────────────────────────")
    metacyc_tsv = download_if_missing(RHEA_METACYC_URL,  rhea_dir / "rhea2metacyc.tsv")
    uniprot_url = RHEA_UNIPROT_ALL_URL if args.use_trembl else RHEA_UNIPROT_URL
    uniprot_fname = "rhea2uniprot.tsv" if args.use_trembl else "rhea2uniprot_sprot.tsv"
    uniprot_tsv = download_if_missing(uniprot_url, rhea_dir / uniprot_fname)

    print("  Parsing Rhea-MetaCyc cross-reference …")
    metacyc_to_rhea = load_rhea_metacyc(metacyc_tsv)
    print(f"    {len(metacyc_to_rhea):,} MetaCyc reactions → Rhea IDs")

    print("  Parsing rhea2uniprot …")
    rhea_to_uniprot = load_rhea_uniprot(uniprot_tsv)
    print(f"    {len(rhea_to_uniprot):,} Rhea reactions → UniProt accessions")

    # ── Protein node → UniProt (from properties_extra.ttl) ────────────────────
    ttl_path = REPO.parent / "data" / "interim" / "properties_extra.ttl"
    print()
    print("── Protein → UniProt mapping ──────────────────────────────────────────────")
    if ttl_path.exists():
        print(f"  Parsing {ttl_path} …")
        local_to_uniprot = parse_uniprot_from_ttl(ttl_path)
        print(f"    {len(local_to_uniprot):,} protein nodes with UniProt accession")
    else:
        print(f"  WARNING: {ttl_path} not found — UniProt mapping unavailable")
        local_to_uniprot = {}

    # ── Load reaction → Rhea from TTL ─────────────────────────────────────────
    rxn_ttl = REPO.parent / "data" / "interim" / "reactions.ttl"
    ttl_rxn_to_rhea: dict[str, set[str]] = {}
    if rxn_ttl.exists():
        print(f"  Parsing {rxn_ttl.name} for embedded Rhea xrefs …")
        ttl_rxn_to_rhea = parse_rhea_from_ttl(rxn_ttl)
        print(f"    {len(ttl_rxn_to_rhea):,} reactions with Rhea xrefs in TTL")

    # ── Load graph and model ───────────────────────────────────────────────────
    print()
    print("── Loading data & model ───────────────────────────────────────────────────")
    set_seed(seed)
    ctx = load_ctx_for_run(hp, str(data_dir), download=False)

    ckpt_files = sorted(run_dir.glob("*_best.pt"))
    assert ckpt_files, f"No *_best.pt checkpoint in {run_dir}"
    ckpt_data = torch.load(ckpt_files[0], weights_only=False)

    model = build_model(
        ctx,
        gnn_name=hp.get("gnn_name", "sage"),
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["num_layers"],
        decoder=hp["decoder"],
        dropout=hp["dropout"],
        random_seed=seed,
        use_norm=hp.get("layer_norm", False),
    )
    model.load_state_dict(ckpt_data["model_state"])
    model.eval()

    # ── Build lookup: reaction_node_idx → Rhea-annotated protein_node_idxs ───
    print()
    print("── Building Rhea lookup ────────────────────────────────────────────────────")
    import pandas as pd
    nodes_df = pd.read_csv(data_dir / "nodes.tsv", sep="\t", low_memory=False)

    # Protein: local_id → node_idx (in PyG order)
    prot_rows = nodes_df[nodes_df.node_type == "Protein"].reset_index(drop=True)
    prot_local_to_idx = {
        iri.split("/")[-1]: i for i, iri in enumerate(prot_rows.node_id)
    }
    # UniProt accession → set of pool protein indices
    uniprot_to_idx: dict[str, set[int]] = defaultdict(set)
    for local_id, accs in local_to_uniprot.items():
        idx = prot_local_to_idx.get(local_id)
        if idx is None:
            continue
        for acc in accs:
            uniprot_to_idx[acc].add(idx)
    print(f"  Proteins with UniProt → pool-idx mapping: {len(uniprot_to_idx):,} accessions "
          f"covering {len({i for s in uniprot_to_idx.values() for i in s}):,} protein nodes")

    # Reaction: node_idx → MetaCyc local_id
    inter_ids   = interaction_node_ids(nodes_df, hp.get("data_type", "raw"))
    inter_local = [iri.split("/")[-1] for iri in inter_ids]

    # Position i is the ctx's Interaction index (Conversion-only for cleaned-graph runs).
    conv_idxs_list = list(range(len(inter_ids)))

    # Map global Interaction-node idx → MetaCyc local id
    inter_idx_to_local = {i: inter_local[i] for i in conv_idxs_list}

    def _rhea_ids_for(local_rxn: str) -> set[str]:
        """Collect Rhea IDs from TTL and from Rhea-MetaCyc TSV.

        MetaCyc local IDs in PlantMetWiki use underscores (RXN_11669) while
        Rhea's rhea-metacyc.tsv uses hyphens (RXN-11669).  Try both forms.
        """
        ids: set[str] = set()
        ids |= ttl_rxn_to_rhea.get(local_rxn, set())
        ids |= metacyc_to_rhea.get(local_rxn, set())
        rhea_form = local_rxn.replace("_", "-")   # RXN_* → RXN-*
        ids |= metacyc_to_rhea.get(rhea_form, set())
        return ids

    # Build: conv_node_idx → set of Rhea-annotated pool protein indices
    rhea_lookup: dict[int, set[int]] = {}

    n_rhea_matched = 0
    for conv_node_idx in conv_idxs_list:
        local_rxn = inter_idx_to_local.get(conv_node_idx, "")
        rhea_ids = _rhea_ids_for(local_rxn)

        if not rhea_ids:
            continue

        # Collect UniProt accessions for these Rhea reactions
        uniprots: set[str] = set()
        for rhea_id in rhea_ids:
            uniprots |= rhea_to_uniprot.get(rhea_id, set())

        # Map UniProt → pool protein indices
        protein_idxs: set[int] = set()
        for acc in uniprots:
            protein_idxs |= uniprot_to_idx.get(acc, set())

        if protein_idxs:
            rhea_lookup[conv_node_idx] = protein_idxs
            n_rhea_matched += 1

    print(f"  Test-set reactions with ≥1 Rhea-annotated pool protein: {n_rhea_matched:,}")
    n_rhea_uniprot = len({acc for ids in rhea_lookup.values() for acc in ids})
    print(f"  Unique Rhea-annotated pool protein indices: {n_rhea_uniprot:,}")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print()
    print("── Results ────────────────────────────────────────────────────────────────")
    splits_to_eval = []
    if args.split in ("val", "both"):
        splits_to_eval.append(("val", ctx.val_data))
    if args.split in ("test", "both"):
        splits_to_eval.append(("test", ctx.test_data))

    header = f"{'Split':<6}  {'Metric':<14}  {'Standard':>12}  {'anyP-H@K':>12}  {'Rhea-val':>12}"
    print(header)
    print("─" * len(header))

    for split_name, split_data in splits_to_eval:
        standard = protein_hits_at_k(
            model.gnn, model.predictor, split_data, ctx,
            k_list=k_list, eval_embedded_only=True, any_catalyst_lookup=None,
        )
        anyp = protein_hits_at_k(
            model.gnn, model.predictor, split_data, ctx,
            k_list=k_list, eval_embedded_only=True,
            any_catalyst_lookup=ctx.all_catalyst_lookup,
        )
        rhea_val = protein_hits_at_k(
            model.gnn, model.predictor, split_data, ctx,
            k_list=k_list, eval_embedded_only=True,
            any_catalyst_lookup=rhea_lookup,
        )
        for k in k_list:
            std_v  = standard.get(f"P-H@{k}", float("nan"))
            any_v  = anyp.get(f"anyP-H@{k}", float("nan"))
            rhea_v = rhea_val.get(f"anyP-H@{k}", float("nan"))
            print(f"{split_name:<6}  P-H@{k:<10}  "
                  f"{std_v*100:>11.2f}%  {any_v*100:>11.2f}%  {rhea_v*100:>11.2f}%")
        n_rhea_eval = rhea_val.get("any_n_eval", 0)
        print(f"         (Rhea evaluated on {n_rhea_eval} unique reactions with Rhea annotations)")
        print()

    # ── Coverage summary ───────────────────────────────────────────────────────
    n_multi_pmw = sum(1 for s in ctx.all_catalyst_lookup.values() if len(s) > 1)
    n_total_pmw = len(ctx.all_catalyst_lookup)
    print(f"PlantMetWiki catalyst lookup : {n_total_pmw:,} reactions, "
          f"{n_multi_pmw:,} ({n_multi_pmw/n_total_pmw*100:.1f}%) with >1 known catalyst")
    print(f"Rhea catalyst lookup         : {len(rhea_lookup):,} reactions with ≥1 pool protein")
    print()
    print("Note: Rhea-val P-H@K is evaluated only on reactions that have Rhea annotations")
    print("and at least one annotated protein in the pool — a subset of the full test set.")
    print("A reaction counted as a 'hit' means the model ranked a Rhea-annotated protein")
    print("in the top-K, regardless of whether PlantMetWiki labels it as a true catalyst.")


if __name__ == "__main__":
    main()
