#!/usr/bin/env python3
"""Package raw embedding inputs into a single distributable archive.

Collects the resolved inputs needed to reproduce all embedding files
without re-running API scraping:

  protein_sequences.tsv     node_id -> amino-acid sequence (8,445 proteins)
  gene_sequences.tsv        node_id -> DNA sequence         (1,359 GeneProducts)
  metabolite_smiles.tsv     node_id -> canonical SMILES     (4,529 Metabolites)
  metabolite_inchikeys.tsv  node_id -> InChIKey             (derived via rdkit)
  conversion_ec.tsv         node_id -> EC number            (13,585 Conversions)
  organism_ncbi.tsv         node_id -> NCBI taxon ID        (424 Organisms)

Usage:
    python scripts/package_embedding_inputs.py [--out PATH]

Default output: data/processed/embedding_inputs.zip
"""
import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR  = ROOT / "data" / "processed"
INTERIM_DIR    = ROOT / "data" / "interim" / "full_embeddings"
NODES_TSV      = PROCESSED_DIR / "nodes.tsv"
DEFAULT_OUT    = PROCESSED_DIR / "embedding_inputs.zip"

NCBITAXON_RE = re.compile(r"NCBITaxon_(\d+)$")


def tsv_bytes(rows: list[tuple], header: list[str]) -> bytes:
    buf = io.StringIO()
    buf.write("\t".join(header) + "\n")
    for row in rows:
        buf.write("\t".join(str(c) if c is not None else "" for c in row) + "\n")
    return buf.getvalue().encode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading nodes.tsv …")
    nodes = pd.read_csv(NODES_TSV, sep="\t", low_memory=False)

    # ── 1. Protein sequences ──────────────────────────────────────────────────
    print("Packing protein_sequences.tsv …")
    with open(INTERIM_DIR / "protein_sequences.json") as f:
        prot_seq: dict[str, str] = json.load(f)
    prot_rows = [(nid, seq) for nid, seq in prot_seq.items() if seq]
    print(f"  {len(prot_rows):,} protein sequences")

    # ── 2. Gene sequences ─────────────────────────────────────────────────────
    print("Packing gene_sequences.tsv …")
    with open(INTERIM_DIR / "gene_sequences.json") as f:
        gene_seq: dict[str, str] = json.load(f)
    gene_rows = [(nid, seq) for nid, seq in gene_seq.items() if seq]
    print(f"  {len(gene_rows):,} gene sequences")

    # ── 3. Metabolite SMILES + InChIKeys ──────────────────────────────────────
    print("Packing metabolite_smiles.tsv and computing InChIKeys …")
    with open(INTERIM_DIR / "metabolite_smiles.json") as f:
        met_smiles: dict[str, str] = json.load(f)

    smiles_rows: list[tuple]     = []
    inchikey_rows: list[tuple]   = []
    n_inchi_fail = 0

    try:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchiKey
        use_rdkit = True
    except ImportError:
        print("  WARNING: rdkit not available — InChIKeys will be empty")
        use_rdkit = False

    for nid, smi in met_smiles.items():
        if not smi:
            continue
        smiles_rows.append((nid, smi))
        if use_rdkit:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                ik = MolToInchiKey(mol)
                inchikey_rows.append((nid, ik or ""))
            else:
                inchikey_rows.append((nid, ""))
                n_inchi_fail += 1
        else:
            inchikey_rows.append((nid, ""))

    print(f"  {len(smiles_rows):,} metabolite SMILES")
    print(f"  {len(inchikey_rows):,} InChIKey rows "
          f"({n_inchi_fail} SMILES not parseable by rdkit)")

    # ── 4. Conversion EC numbers ──────────────────────────────────────────────
    print("Packing conversion_ec.tsv …")
    conv_mask = (
        (nodes["node_type"] == "Interaction") &
        (nodes["interaction_subtype"] == "Conversion") &
        nodes["ec_number"].notna()
    )
    conv_df = nodes.loc[conv_mask, ["node_id", "ec_number"]]
    ec_rows = list(conv_df.itertuples(index=False, name=None))
    print(f"  {len(ec_rows):,} Conversion nodes with EC annotation "
          f"(of {int((nodes['interaction_subtype'] == 'Conversion').sum()):,} total)")

    # ── 5. Organism NCBI IDs ─────────────────────────────────────────────────
    print("Packing organism_ncbi.tsv …")
    org_nodes = nodes.loc[nodes["node_type"] == "Organism", "node_id"]
    org_rows: list[tuple] = []
    for nid in org_nodes:
        m = NCBITAXON_RE.search(nid)
        org_rows.append((nid, m.group(1) if m else ""))
    print(f"  {len(org_rows):,} organism nodes")

    # ── Write archive ─────────────────────────────────────────────────────────
    print(f"\nWriting {args.out} …")
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("protein_sequences.tsv",
                    tsv_bytes(prot_rows, ["node_id", "sequence"]).decode())
        zf.writestr("gene_sequences.tsv",
                    tsv_bytes(gene_rows, ["node_id", "sequence"]).decode())
        zf.writestr("metabolite_smiles.tsv",
                    tsv_bytes(smiles_rows, ["node_id", "smiles"]).decode())
        zf.writestr("metabolite_inchikeys.tsv",
                    tsv_bytes(inchikey_rows, ["node_id", "inchikey"]).decode())
        zf.writestr("conversion_ec.tsv",
                    tsv_bytes(ec_rows, ["node_id", "ec_number"]).decode())
        zf.writestr("organism_ncbi.tsv",
                    tsv_bytes(org_rows, ["node_id", "ncbi_taxon_id"]).decode())

        # README inside the zip
        readme = f"""# Embedding inputs for PlantMetBench

Resolved inputs needed to reproduce all embedding files without re-running
API scraping (UniProt, NCBI, PubChem, MetaNetX).

## Files

| File | Description | Rows |
|------|-------------|------|
| protein_sequences.tsv  | node_id → amino-acid sequence       | {len(prot_rows):,} |
| gene_sequences.tsv     | node_id → DNA sequence (GenBank)    | {len(gene_rows):,} |
| metabolite_smiles.tsv  | node_id → canonical SMILES          | {len(smiles_rows):,} |
| metabolite_inchikeys.tsv | node_id → InChIKey (rdkit-derived) | {len(inchikey_rows):,} |
| conversion_ec.tsv      | node_id → EC number                 | {len(ec_rows):,} |
| organism_ncbi.tsv      | node_id → NCBI taxon ID             | {len(org_rows):,} |

## How these were generated

- **protein_sequences**: resolved via UniProt REST API / RCSB PDB
  (scripts/learnathon_embed_proteins.py / scripts/fulldata_embed_proteins.py)
- **gene_sequences**: resolved via NCBI nuccore/eutils
  (scripts/learnathon_embed_genes.py)
- **metabolite_smiles**: resolved via PubChem, MetaNetX, KEGG
  (scripts/learnathon_embed_metabolites.py)
- **metabolite_inchikeys**: derived from SMILES using rdkit
- **conversion_ec**: parsed from data/interim/reactions.ttl via rdflib
  (scripts/compute_ec_embeddings.py)
- **organism_ncbi**: extracted from data/processed/nodes.tsv (NCBITaxon IRI)
"""
        zf.writestr("README.md", readme)

    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"Done. Archive: {args.out}  ({size_mb:.1f} MB)")
    print("\nContents:")
    with zipfile.ZipFile(args.out) as zf:
        for info in zf.infolist():
            print(f"  {info.filename:<35}  {info.file_size/1024:>8.0f} KB")


if __name__ == "__main__":
    main()
