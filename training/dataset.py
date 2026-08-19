#!/usr/bin/env python3
"""Download and load the PlantMetBench dataset from Zenodo.

Full dataset (424 plant species, taxa-holdout split):
    DOI: 10.5281/zenodo.20847651

The concept DOI always resolves to the latest version; individual version DOIs
are printed on first download so you can record exactly which version was used.

Typical usage:
    from dataset import load_data
    ctx = load_data()               # auto-downloads to data/ if not present
    ctx = load_data(device="cuda")
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# ── Constants ─────────────────────────────────────────────────────────────────
ZENODO_CONCEPT_DOI   = "10.5281/zenodo.20847651"
ZENODO_CONCEPT_RECID = "20847651"
DATA_DIR_DEFAULT     = Path(__file__).parent.parent / "data"

TARGET_EDGE = ("Protein", "catalyzes", "Interaction")
REV_EDGE    = ("Interaction", "catalyzed_by", "Protein")

# InChIKey suffixes / ChEBI IDs for currency/hub metabolites (>100 Conversions).
_CURRENCY_SUFFIXES: frozenset[str] = frozenset({
    "GPRLSGONYQIRFK-UHFFFAOYSA-N",   # ATP
    "XPPKVPWEQAFLFU-UHFFFAOYSA-K",   # ADP
    "GVVPGTZRZFNKDS-JXMROGBWSA-K",   # ATP alt form
    "XLYOFNOQVPJJNP-UHFFFAOYSA-N",   # H2O
    "CURLTUGMZLYLDI-UHFFFAOYSA-N",   # CO2
    "XCCTYIAWTASOJW-XVFCMESISA-K",   # NAD+
    "RGJOEKWQDUBAIZ-IBOSZNHHSA-J",   # NADH
    "XTWYTFMLZFPYCI-KQYNXXCUSA-K",   # NADP+
    "ACFIXJIJDZMPPO-NNYOXOHSSA-J",   # NADPH
    "CHEBI:17499",                    # NADPH (ChEBI)
    "NBIIXXVUZAFLBC-UHFFFAOYSA-L",   # Pi
    "ZJUKTBDSGOFHSH-WFMPWKQPSA-N",   # CoA alt form
    "BOPGDPNILDQYTO-NNYOXOHSSA-L",   # CoA-SH
    "CHEBI:13392",                    # CoA (ChEBI)
    "VWFJDQUYCIWHTN-YFVJMOTDSA-K",   # Acetyl-CoA
    "CHEBI:15339",                    # Acetyl-CoA (ChEBI)
    "UDMBCSSLTHHNCD-KQYNXXCUSA-L",   # FAD
    "XJLXINKUBYWONI-NNYOXOHSSA-K",   # FADH2
    "WHUUTDBJXJRKMK-VKHMYHEASA-M",   # SAM
    "HSCJRCZFDFQWRP-JZMIEXBBSA-L",   # SAH
    "MHAJPDPJQMAIIY-UHFFFAOYSA-N",   # H2O2
    "QGZKDVFQNNGYKY-UHFFFAOYSA-O",   # NH4+
})


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class GraphContext:
    """Everything a training/eval loop needs, loaded once per process.

    Attributes
    ----------
    train_data, val_data, test_data : HeteroData objects on ctx.device
    target_edge  : ("Protein", "catalyzes", "Interaction")
    src_type     : "Protein"
    dst_type     : "Interaction"
    nodes_df     : DataFrame with node_id, node_type, interaction_subtype, ...
    conv_idxs    : list of Interaction-node indices that are Conversion subtype
    conv_idxs_tensor : conv_idxs as a LongTensor
    embedded_protein_mask : bool tensor [n_proteins] — True where the protein
                   has an ESM embedding (None if embedded_only_ranking=False)
    all_catalyst_lookup   : {conv_idx -> set of protein_idxs} across all splits
    device       : torch.device
    data_dir     : Path where data files are stored
    split_ei     : {"train": tensor, "val": tensor, "test": tensor} positive edge indices
    """
    train_data: object
    val_data: object
    test_data: object
    target_edge: tuple
    src_type: str
    dst_type: str
    nodes_df: pd.DataFrame
    conv_idxs: list
    conv_idxs_tensor: torch.Tensor
    embedded_protein_mask: Optional[torch.Tensor]
    all_catalyst_lookup: dict
    device: torch.device
    data_dir: Path
    split_ei: dict


# ── Zenodo download ───────────────────────────────────────────────────────────

def download_data(data_dir: Path = DATA_DIR_DEFAULT, force: bool = False) -> None:
    """Download all files from the latest Zenodo record for ZENODO_CONCEPT_DOI.

    Files that already exist with a matching MD5 checksum are skipped.
    Pass force=True to re-download unconditionally.
    """
    try:
        import requests
    except ImportError:
        sys.exit("Missing dependency: pip install requests")

    data_dir.mkdir(parents=True, exist_ok=True)

    api_url = f"https://zenodo.org/api/records/{ZENODO_CONCEPT_RECID}"
    print(f"Fetching Zenodo record metadata …  ({api_url})")
    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        sys.exit(f"Could not reach Zenodo API: {exc}")

    record  = resp.json()
    version = record.get("metadata", {}).get("version", "?")
    rec_doi = record.get("doi", ZENODO_CONCEPT_DOI)
    print(f"  Record: {rec_doi}  (version {version})")
    print(f"  Please record this version in your paper / run config.")

    files = record.get("files", [])
    if not files:
        sys.exit(
            "No files found in this Zenodo record. "
            "If the record is not yet public, upload the dataset first."
        )

    for f in files:
        name         = f["key"]
        size         = f.get("size", 0)
        checksum_str = f.get("checksum", "")
        dl_url       = f["links"]["self"]
        dest         = data_dir / name

        if not force and dest.exists():
            if checksum_str.startswith("md5:") and _md5(dest) == checksum_str[4:]:
                print(f"  {name}: OK (already downloaded, checksum verified)")
                continue
            elif not checksum_str:
                print(f"  {name}: OK (already downloaded, no checksum in record)")
                continue
            else:
                print(f"  {name}: checksum mismatch — re-downloading")

        mb = size / 1e6
        print(f"  Downloading {name}  ({mb:.1f} MB) …")
        _download_file(dl_url, dest)
        print(f"    → {dest}")

    print("Download complete.\n")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path) -> None:
    import requests
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(
    data_dir: str | Path = DATA_DIR_DEFAULT,
    device: str | torch.device | None = None,
    neg_ratio: float = 1.0,
    random_seed: int = 42,
    remove_is_part_of: bool = True,
    embedded_only_ranking: bool = True,
    disjoint_train_ratio: float = 0.2,
    keep_catalyzed_by: bool = False,
    remove_currency_metabolites: bool = False,
    remove_all_metabolites: bool = False,
    load_ec_embeddings: bool = False,
    remove_gene_organism_edges: bool = False,
    organism_embeddings_path: str | Path | None = None,
    organism_embedding_type: str = "mds",
    download: bool = True,
) -> GraphContext:
    """Load the PlantMetBench dataset and return a GraphContext ready for training.

    Parameters
    ----------
    data_dir : directory containing heterodata.pt, splits_taxa.pt, nodes.tsv
               (auto-downloaded from Zenodo if files are missing)
    device   : "cuda" / "cpu" (default: auto-select GPU if available)
    neg_ratio : random negatives per positive edge in val/test (default 1.0)
    random_seed : for reproducible negative sampling and disjoint split
    remove_is_part_of : drop Pathway co-membership edges from the MP graph.
        Recommended: True. Without this, Pathway membership is a trivial shortcut
        that inflates AUC without any biological learning.
    embedded_only_ranking : restrict the P-H@K ranking pool to proteins with
        ESM embeddings (~8,445 of ~13,682). Recommended: True.
    disjoint_train_ratio : fraction of training positives held out from the MP
        graph (supervision-only). Default 0.2 prevents the 1-hop shortcut.
    keep_catalyzed_by : if True, add (Interaction, catalyzed_by, Protein) reverse
        edge to the MP graph. Default False — this creates a 2-hop shortcut.
    remove_currency_metabolites : remove edges involving ATP, ADP, H2O, NAD+,
        NADH, NADP+, NADPH, CoA, FAD, FADH2, Pi, CO2, H2O2, SAM, SAH, NH4+.
        These hub nodes appear in >100 Conversions and can shortcut discrimination.
    remove_all_metabolites : remove the entire Metabolite node store. Reaction
        chemistry is then represented solely by conversion fingerprints.
        Mutually exclusive with remove_currency_metabolites.
    load_ec_embeddings : append 237-dim EC hierarchy one-hot to Interaction
        features (3072→3309-dim). WARNING: causes ~80% val→test gaps in the
        taxa holdout — use only as an ablation, not as the baseline.
    remove_gene_organism_edges : remove (GeneProduct, organism, Organism) edges
        while keeping (Protein, organism, Organism).
    organism_embeddings_path : path to embeddings_organism.pt. Replaces the
        random 64-dim Organism features with taxonomy-aware MDS coordinates.
    organism_embedding_type : "mds" (64-dim) or "multihot" (702-dim lineage).
    download : auto-download from Zenodo if required files are absent.
    """
    data_dir = Path(data_dir)
    device   = torch.device(device) if device else \
               torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _check_or_download(data_dir, download)

    # ── Raw graph ─────────────────────────────────────────────────────────────
    print(f"Loading graph from {data_dir} …")
    data     = torch.load(data_dir / "heterodata.pt", weights_only=False)
    nodes_df = pd.read_csv(data_dir / "nodes.tsv", sep="\t", low_memory=False)

    torch.manual_seed(random_seed)
    for nt in data.node_types:
        if not hasattr(data[nt], "x") or data[nt].x is None:
            data[nt].x = torch.randn(data[nt].num_nodes, 64)

    # ── Organism embeddings (optional taxonomy-aware features) ────────────────
    if organism_embeddings_path is not None:
        org_emb_path = Path(organism_embeddings_path)
        payload = torch.load(org_emb_path, weights_only=False)
        _key = "lineage_multihot" if organism_embedding_type == "multihot" else organism_embedding_type
        mds         = payload[_key]
        taxid_order = payload["taxid_order"]
        tid2row     = {tid: i for i, tid in enumerate(taxid_order)}
        import re as _re
        org_ids     = nodes_df.loc[nodes_df["node_type"] == "Organism", "node_id"].tolist()
        n_org       = data["Organism"].num_nodes
        new_dim     = mds.shape[1]
        org_x       = torch.zeros(n_org, new_dim)
        n_replaced  = 0
        for local_idx, nid in enumerate(org_ids):
            m = _re.search(r"NCBITaxon_(\d+)", nid)
            if m:
                row = tid2row.get(m.group(1))
                if row is not None:
                    org_x[local_idx] = mds[row].to(device)
                    n_replaced += 1
        data["Organism"].x = org_x
        print(f"  Organism features: replaced {n_replaced}/{n_org} nodes with "
              f"taxonomy {organism_embedding_type} ({new_dim}-dim)")

    # ── Protein embeddings (load from embeddings_protein.pt if present) ───────
    prot_emb_path = data_dir / "embeddings_protein.pt"
    if prot_emb_path.exists():
        prot_emb: dict[str, torch.Tensor] = torch.load(prot_emb_path, weights_only=False)
        emb_dim  = next(iter(prot_emb.values())).shape[0]
        prot_ids = nodes_df.loc[nodes_df["node_type"] == "Protein", "node_id"].tolist()
        n_prot   = len(prot_ids)
        feat     = torch.zeros(n_prot, emb_dim, dtype=torch.float32)
        n_covered = sum(1 for nid in prot_ids if nid in prot_emb)
        for i, nid in enumerate(prot_ids):
            if nid in prot_emb:
                feat[i] = prot_emb[nid]
        data["Protein"].x = feat
        print(f"  Protein embeddings: {n_covered:,}/{n_prot:,} nodes covered "
              f"({n_covered/n_prot:.1%})  dim={emb_dim}")

    # ── Pathway shortcut removal ──────────────────────────────────────────────
    if remove_is_part_of:
        dropped = [et for et in list(data.edge_types) if et[1] == "is_part_of"]
        for et in dropped:
            del data[et]
        for et in [et for et in list(data.edge_types) if "Pathway" in (et[0], et[2])]:
            del data[et]
        if "Pathway" in data.node_types:
            del data["Pathway"]
        print(f"  Removed {len(dropped)} is_part_of edge type(s) and Pathway node store")

    if remove_all_metabolites:
        met_edges = [et for et in list(data.edge_types) if "Metabolite" in (et[0], et[2])]
        for et in met_edges:
            del data[et]
        if "Metabolite" in data.node_types:
            del data["Metabolite"]
        print(f"  Removed {len(met_edges)} Metabolite edge type(s) and Metabolite node store")

    if remove_gene_organism_edges:
        gene_org_key = ("GeneProduct", "organism", "Organism")
        if gene_org_key in data.edge_types:
            del data[gene_org_key]
            print("  Removed (GeneProduct, organism, Organism) edges")

    if remove_currency_metabolites:
        met_ids = nodes_df.loc[nodes_df["node_type"] == "Metabolite", "node_id"].tolist()
        currency_idx = [
            i for i, nid in enumerate(met_ids)
            if nid.rsplit("/", 1)[-1] in _CURRENCY_SUFFIXES
        ]
        if currency_idx:
            currency_tensor = torch.tensor(currency_idx, dtype=torch.long)
            n_removed = 0
            for et in list(data.edge_types):
                src_type, _rel, dst_type = et
                ei = data[et].edge_index
                if dst_type == "Metabolite":
                    keep = ~torch.isin(ei[1], currency_tensor)
                elif src_type == "Metabolite" and dst_type != "Organism":
                    keep = ~torch.isin(ei[0], currency_tensor)
                else:
                    continue
                n = int((~keep).sum())
                if n > 0:
                    data[et].edge_index = ei[:, keep]
                    n_removed += n
            print(f"  Blocked {len(currency_idx)} currency metabolite nodes, "
                  f"removed {n_removed:,} edges")

    # ── Taxa-holdout split ────────────────────────────────────────────────────
    print("Loading taxa split …")
    splits_taxa = torch.load(data_dir / "splits_taxa.pt", weights_only=False)
    assert tuple(splits_taxa["target_edge"]) == TARGET_EDGE
    pos_ei = {name: splits_taxa[f"{name}_edge_index"]
              for name in ("train", "val", "test")}

    n_src    = data[TARGET_EDGE[0]].num_nodes
    all_pairs = {
        (int(s), int(d))
        for ei in pos_ei.values()
        for s, d in zip(ei[0].tolist(), ei[1].tolist())
    }
    rng = np.random.default_rng(random_seed)

    _inter_ids    = nodes_df.loc[nodes_df["node_type"] == "Interaction", "node_id"].tolist()
    _inter_id2idx = {nid: i for i, nid in enumerate(_inter_ids)}
    _conv_neg_ids = nodes_df.loc[
        (nodes_df["node_type"] == "Interaction") &
        (nodes_df["interaction_subtype"] == "Conversion"),
        "node_id",
    ].tolist()
    conv_neg_idxs = np.array([_inter_id2idx[nid] for nid in _conv_neg_ids], dtype=np.int64)

    def _sample_negs(n: int) -> torch.Tensor:
        out = []
        for _ in range(200):
            ss = rng.integers(0, n_src, size=n - len(out))
            dd = conv_neg_idxs[rng.integers(0, len(conv_neg_idxs), size=n - len(out))]
            out += [(int(s), int(d)) for s, d in zip(ss, dd)
                    if (int(s), int(d)) not in all_pairs]
            if len(out) >= n:
                break
        if len(out) < n:
            raise RuntimeError(f"Could not sample {n} negatives after 200 passes.")
        return torch.tensor(out[:n], dtype=torch.long).T

    # Disjoint split: 20% of training positives are supervision-only (absent from MP graph).
    n_train = pos_ei["train"].shape[1]
    n_mp    = int(round(n_train * (1.0 - disjoint_train_ratio)))
    perm_tr = torch.randperm(n_train, generator=torch.Generator().manual_seed(random_seed))
    train_mp  = pos_ei["train"][:, perm_tr[:n_mp]]
    train_sup = pos_ei["train"][:, perm_tr[n_mp:]]

    split_data = {}
    for name in ("train", "val", "test"):
        sup_ei = train_sup if name == "train" else pos_ei[name]
        n_pos  = sup_ei.shape[1]
        neg    = _sample_negs(int(round(n_pos * neg_ratio)))
        eli    = torch.cat([sup_ei, neg], dim=1)
        lbl    = torch.cat([torch.ones(n_pos), torch.zeros(neg.shape[1])])
        perm   = torch.randperm(eli.shape[1], generator=torch.Generator().manual_seed(random_seed))
        d      = data.clone()
        d[TARGET_EDGE].edge_index       = train_mp
        if keep_catalyzed_by:
            d[REV_EDGE].edge_index      = train_mp.flip(0)
        elif REV_EDGE in d.edge_types:
            del d[REV_EDGE]
        d[TARGET_EDGE].edge_label_index = eli[:, perm]
        d[TARGET_EDGE].edge_label       = lbl[perm]
        split_data[name] = d.to(device)

    # ── Conversion fingerprints (MAP4-based, ~3072-dim) ───────────────────────
    inter_ids = nodes_df.loc[nodes_df["node_type"] == "Interaction", "node_id"].tolist()
    n_inter   = len(inter_ids)

    conv_emb_path = data_dir / "embeddings_conversion.pt"
    if conv_emb_path.exists():
        conv_emb: dict[str, torch.Tensor] = torch.load(conv_emb_path, weights_only=False)
        emb_dim = next(iter(conv_emb.values())).shape[0]
        feat = torch.zeros(n_inter, emb_dim, dtype=torch.float32)
        n_conv_covered = sum(1 for nid in inter_ids if nid in conv_emb)
        for i, nid in enumerate(inter_ids):
            if nid in conv_emb:
                feat[i] = conv_emb[nid]
        for split_name in ("train", "val", "test"):
            split_data[split_name]["Interaction"].x = feat.to(device)
        print(f"  Conversion fingerprints: {n_conv_covered:,}/{n_inter:,} nodes "
              f"({n_conv_covered/n_inter:.1%})  dim={emb_dim}")

    # ── EC embeddings (ablation only — causes val→test gap) ───────────────────
    ec_emb_path = data_dir / "embeddings_ec.pt"
    if load_ec_embeddings and ec_emb_path.exists():
        ec_emb: dict[str, torch.Tensor] = torch.load(ec_emb_path, weights_only=False)
        ec_dim = next(iter(ec_emb.values())).shape[0]
        ec_feat = torch.zeros(n_inter, ec_dim, dtype=torch.float32)
        n_ec_covered = sum(1 for nid in inter_ids if nid in ec_emb)
        for i, nid in enumerate(inter_ids):
            if nid in ec_emb:
                ec_feat[i] = ec_emb[nid]
        for split_name in ("train", "val", "test"):
            existing = split_data[split_name]["Interaction"].x
            split_data[split_name]["Interaction"].x = torch.cat(
                [existing, ec_feat.to(device)], dim=-1
            )
        print(f"  EC embeddings (ablation): {n_ec_covered:,}/{n_inter:,}  dim={ec_dim}")

    # ── Ranking pool ──────────────────────────────────────────────────────────
    id2idx = {
        ntype: {nid: i for i, nid in enumerate(sub["node_id"].tolist())}
        for ntype, sub in nodes_df.groupby("node_type")
    }
    conv_ids = nodes_df.loc[
        (nodes_df["node_type"] == "Interaction") &
        (nodes_df["interaction_subtype"] == "Conversion"),
        "node_id",
    ].tolist()
    conv_idxs        = [id2idx["Interaction"][nid] for nid in conv_ids]
    conv_idxs_tensor = torch.tensor(conv_idxs, dtype=torch.long)

    emb_mask = _build_embedded_mask_from_x(data, nodes_df) if embedded_only_ranking else None

    all_catalyst_lookup: dict[int, set] = {}
    for ei in pos_ei.values():
        for p, c in zip(ei[0].tolist(), ei[1].tolist()):
            all_catalyst_lookup.setdefault(int(c), set()).add(int(p))

    ctx = GraphContext(
        train_data=split_data["train"],
        val_data=split_data["val"],
        test_data=split_data["test"],
        target_edge=TARGET_EDGE,
        src_type=TARGET_EDGE[0],
        dst_type=TARGET_EDGE[2],
        nodes_df=nodes_df,
        conv_idxs=conv_idxs,
        conv_idxs_tensor=conv_idxs_tensor,
        embedded_protein_mask=emb_mask,
        all_catalyst_lookup=all_catalyst_lookup,
        device=device,
        data_dir=data_dir,
        split_ei=pos_ei,
    )
    _print_summary(ctx, pos_ei, embedded_only_ranking)
    return ctx


def _check_or_download(data_dir: Path, download: bool) -> None:
    required = ["heterodata.pt", "splits_taxa.pt", "nodes.tsv"]
    missing  = [f for f in required if not (data_dir / f).exists()]
    if missing:
        if download:
            print(f"Missing data files: {missing}")
            download_data(data_dir)
            for zip_path in data_dir.glob("*.zip"):
                print(f"Extracting {zip_path.name} …")
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(data_dir)
        else:
            sys.exit(
                f"Data files missing from {data_dir}: {missing}\n"
                f"Run dataset.download_data() or pass download=True to load_data()."
            )


def _build_embedded_mask_from_x(data, nodes_df: pd.DataFrame) -> torch.Tensor:
    x = data["Protein"].x
    if x is None:
        n = nodes_df[nodes_df["node_type"] == "Protein"].shape[0]
        print("  WARNING: Protein.x is None — treating all proteins as embedded")
        return torch.ones(n, dtype=torch.bool)
    mask = x.norm(dim=-1) > 0
    n    = int(mask.sum())
    print(f"  Ranking pool: {n:,}/{mask.shape[0]:,} proteins have embeddings "
          f"({n/mask.shape[0]:.1%})")
    return mask


def _print_summary(ctx: GraphContext, pos_ei: dict, embedded_only: bool) -> None:
    print()
    print("─" * 72)
    print("Dataset summary")
    print("─" * 72)
    print(f"  Target edge : {ctx.target_edge}")
    print()
    print("Node types:")
    for nt in ctx.train_data.node_types:
        store = ctx.train_data[nt]
        x_shape = tuple(store.x.shape) if "x" in store else None
        print(f"  {nt:<12}  num_nodes={store.num_nodes:>9,d}  x={x_shape}")
    print("Edge types:")
    for et in ctx.train_data.edge_types:
        store = ctx.train_data[et]
        print(f"  {str(et):<50}  num_edges={store.num_edges:>10,d}")
    print()
    print("  Positive edge splits:")
    for name, ei in pos_ei.items():
        print(f"    {name:6s} {ei.shape[1]:>7,} edges")
    pool_n = int(ctx.embedded_protein_mask.sum()) if ctx.embedded_protein_mask is not None \
             else ctx.train_data[ctx.src_type].num_nodes
    print()
    print(f"  Ranking pool : {pool_n:,} proteins  "
          f"({'embedded only' if embedded_only else 'all'})")
    print(f"  Random P-H@50: {50/pool_n:.4f}  ({50/pool_n:.2%})")
    print("─" * 72)
    print()
