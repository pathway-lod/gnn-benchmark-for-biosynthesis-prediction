#!/usr/bin/env python3
"""
PCA / t-SNE / UMAP projections of PlantMetBench node embeddings, laid out as
a 2x2 grid (PCA, t-SNE, UMAP, shared legend in the 4th slot) rather than a
cramped single row of three panels -- larger fonts, larger legend, and
explicit tick numbers on every projection axis.

Generates all five embedding figures referenced in the appendix:
  embedding_split_protein.png      ESM-C protein embeddings, by taxa-split membership
  embedding_organism_protein.png   ESM-C protein embeddings, by organism
  embedding_organism_geneproduct.png  PlantCaduceus gene embeddings, by organism
  embedding_organism_metabolite.png   MAP4 metabolite embeddings, by organism
  embedding_ecclass_conversion.png    MAP4 DRFP-approx reaction fingerprints, by EC top-level class

Usage (from repo root):
    python scripts/plot_embeddings.py                 # all five
    python scripts/plot_embeddings.py --only split_protein organism_protein
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

RNG_SEED = 42
TOP_N_ORGANISMS = 10

EC_NAMES = {
    1: "Oxidoreductases", 2: "Transferases", 3: "Hydrolases", 4: "Lyases",
    5: "Isomerases", 6: "Ligases", 7: "Translocases",
}

# Muted, colour-blind-friendlier categorical palette (tab10-derived) plus a
# neutral grey for "other" / unlabelled categories.
PALETTE = ["#3B70A3", "#C2782E", "#3D8A5C", "#A24552", "#7A5DA8",
           "#6B4A3A", "#B0658C", "#5A7A8C", "#8A8A3D", "#4E7FBF"]
OTHER_COLOR = "#C7CCD1"


# ── Dimensionality reduction ────────────────────────────────────────────────
def reduce_all(X: np.ndarray) -> dict[str, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap

    print(f"    PCA  ({X.shape[0]:,} x {X.shape[1]})…")
    pca = PCA(n_components=2, random_state=RNG_SEED).fit_transform(X)
    print(f"    t-SNE …")
    tsne = TSNE(n_components=2, random_state=RNG_SEED, init="pca",
                perplexity=min(30, max(5, X.shape[0] // 100))).fit_transform(X)
    print(f"    UMAP …")
    um = umap.UMAP(n_components=2, random_state=RNG_SEED).fit_transform(X)
    return {"PCA": pca, "t-SNE": tsne, "UMAP": um}


# ── Shared 2x2 plot (PCA, t-SNE, UMAP, legend) ──────────────────────────────
def make_figure(projections: dict[str, np.ndarray], labels: np.ndarray,
                 categories: list[str], colors: dict[str, str],
                 subject: str, out_name: str, point_size: float = 6.0):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.lines as mlines

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 16,
        "axes.linewidth": 0.8,
        "xtick.major.size": 3.5, "xtick.major.width": 0.7,
        "ytick.major.size": 3.5, "ytick.major.width": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(12.0, 13.4))
    gs = gridspec.GridSpec(2, 2, left=0.08, right=0.97, bottom=0.055, top=0.89,
                            wspace=0.28, hspace=0.34)
    axes = {
        "PCA": fig.add_subplot(gs[0, 0]),
        "t-SNE": fig.add_subplot(gs[0, 1]),
        "UMAP": fig.add_subplot(gs[1, 0]),
    }
    ax_legend = fig.add_subplot(gs[1, 1])
    ax_legend.axis("off")

    for name, ax in axes.items():
        emb = projections[name]
        # plot "other"/background categories first so highlighted ones sit on top
        order = sorted(range(len(categories)),
                        key=lambda i: 0 if categories[i] in ("Other", "other") else 1)
        for i in order:
            cat = categories[i]
            mask = labels == cat
            if not mask.any():
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1], s=point_size, color=colors[cat],
                       alpha=0.55 if cat.lower().startswith("other") else 0.8,
                       linewidths=0, zorder=1 if cat.lower().startswith("other") else 2)
        ax.set_title(name, fontsize=19, color="#1A1E2A", pad=10)
        ax.tick_params(labelsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D0D8E0")
        ax.spines["bottom"].set_color("#D0D8E0")
        ax.tick_params(colors="#5A6070", length=3.5)
        ax.set_xlabel(f"{name} 1", fontsize=14.5)
        ax.set_ylabel(f"{name} 2", fontsize=14.5)

    fig.suptitle(subject, fontsize=21, color="#1A1E2A", y=0.985)

    handles = [mlines.Line2D([], [], marker="o", linestyle="", markersize=11,
                              color=colors[c], label=c) for c in categories]
    ax_legend.legend(handles=handles, loc="center left", fontsize=15.5,
                      frameon=True, framealpha=0.95, edgecolor="#D8DDE4",
                      handletextpad=0.8, labelspacing=0.9, borderpad=1.0)

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{out_name}.{ext}"
        fig.savefig(out, dpi=250, bbox_inches="tight")
        print(f"  Saved → {out}")
    plt.close(fig)


def top_n_plus_other(node_ids: list[str], org_idx: np.ndarray,
                      org_name: dict[int, str], n: int = TOP_N_ORGANISMS):
    """Return per-node category labels: top-N organisms by count (named),
    everything else as 'Other'."""
    counts = pd.Series(org_idx).value_counts()
    top_orgs = counts.head(n).index.tolist()
    cat_name = {o: org_name.get(o, f"Organism {o}") for o in top_orgs}
    labels = np.array([cat_name.get(o, "Other") for o in org_idx])
    categories = [cat_name[o] for o in top_orgs] + ["Other"]
    colors = {cat_name[o]: PALETTE[i % len(PALETTE)] for i, o in enumerate(top_orgs)}
    colors["Other"] = OTHER_COLOR
    return labels, categories, colors


# ── Per-figure data loaders ─────────────────────────────────────────────────
def load_organism_names(taxids: list[str]) -> dict[str, str]:
    import json
    cache_path = DATA_DIR / "organism_names_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = [t for t in taxids if t not in cache]
    if missing:
        import time
        import xml.etree.ElementTree as ET
        import requests
        print(f"  Resolving {len(missing)} organism name(s) via NCBI efetch …")
        for i in range(0, len(missing), 50):
            batch = missing[i:i + 50]
            r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                              params={"db": "taxonomy", "id": ",".join(batch), "retmode": "xml"},
                              timeout=60)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            for taxon in root.findall("Taxon"):
                tid = taxon.findtext("TaxId", "").strip()
                sci = taxon.findtext("ScientificName", "").strip()
                common = taxon.findtext("OtherNames/GenbankCommonName", "").strip()
                cache[tid] = f"{sci} ({common})" if common and common.lower() != sci.lower() else sci
            time.sleep(0.34)
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    return cache


def taxid_from_node_id(node_id: str) -> str | None:
    m = re.search(r"NCBITaxon_(\d+)", node_id)
    return m.group(1) if m else None


def fig_split_protein():
    data = torch.load(DATA_DIR / "heterodata.pt", weights_only=False)
    nodes = pd.read_csv(DATA_DIR / "nodes.tsv", sep="\t", low_memory=False)
    prot_ids = nodes.loc[nodes.node_type == "Protein", "node_id"].tolist()
    prot_emb = torch.load(DATA_DIR / "embeddings_protein.pt", weights_only=False)

    embedded_idx = [i for i, nid in enumerate(prot_ids) if nid in prot_emb]
    X = np.stack([prot_emb[prot_ids[i]].numpy() for i in embedded_idx])

    # A protein's split membership is defined by which split's *catalyzes*
    # edges it actually appears as source in -- not by its organism's split
    # assignment, since nearly every protein has some organism edge and that
    # would make almost none of them fall into "(not a target-edge protein)".
    splits = torch.load(DATA_DIR / "splits_taxa.pt", weights_only=False)
    prot_split: dict[int, str] = {}
    for split_name, key in (("train", "train_edge_index"), ("val", "val_edge_index"),
                             ("test", "test_edge_index")):
        for p in splits[key][0].tolist():
            prot_split[p] = split_name

    def split_of(i):
        return prot_split.get(i, "(not a target-edge protein)")

    labels = np.array([split_of(i) for i in embedded_idx])
    categories = ["(not a target-edge protein)", "train", "val", "test"]
    colors = {"(not a target-edge protein)": OTHER_COLOR, "train": PALETTE[0],
              "val": PALETTE[1], "test": PALETTE[2]}
    return X, labels, categories, colors, f"ESM-C protein embeddings by taxa-split membership ({len(embedded_idx):,} proteins)"


def fig_organism_protein():
    data = torch.load(DATA_DIR / "heterodata.pt", weights_only=False)
    nodes = pd.read_csv(DATA_DIR / "nodes.tsv", sep="\t", low_memory=False)
    prot_ids = nodes.loc[nodes.node_type == "Protein", "node_id"].tolist()
    prot_emb = torch.load(DATA_DIR / "embeddings_protein.pt", weights_only=False)

    embedded_idx = [i for i, nid in enumerate(prot_ids) if nid in prot_emb]
    X = np.stack([prot_emb[prot_ids[i]].numpy() for i in embedded_idx])

    po_ei = data[("Protein", "organism", "Organism")].edge_index
    prot_to_org = {}
    for p, o in zip(po_ei[0].tolist(), po_ei[1].tolist()):
        prot_to_org.setdefault(p, o)
    org_idx = np.array([prot_to_org.get(i, -1) for i in embedded_idx])

    org_ids = nodes.loc[nodes.node_type == "Organism", "node_id"].tolist()
    counts = pd.Series(org_idx).value_counts()
    top_taxids = [taxid_from_node_id(org_ids[o]) for o in counts.head(TOP_N_ORGANISMS).index if o >= 0]
    names_by_taxid = load_organism_names([t for t in top_taxids if t])
    org_name = {o: names_by_taxid.get(taxid_from_node_id(org_ids[o]), org_ids[o])
                for o in counts.head(TOP_N_ORGANISMS).index if o >= 0}

    labels, categories, colors = top_n_plus_other(prot_ids, org_idx, org_name)
    return X, labels, categories, colors, f"ESM-C protein embeddings by organism ({len(embedded_idx):,} proteins)"


def fig_organism_geneproduct():
    data = torch.load(DATA_DIR / "heterodata.pt", weights_only=False)
    nodes = pd.read_csv(DATA_DIR / "nodes.tsv", sep="\t", low_memory=False)
    x = data["GeneProduct"].x.numpy()
    nonzero = np.where(np.abs(x).sum(1) > 0)[0]
    X = x[nonzero]

    go_ei = data[("GeneProduct", "organism", "Organism")].edge_index
    gp_to_org = {}
    for g, o in zip(go_ei[0].tolist(), go_ei[1].tolist()):
        gp_to_org.setdefault(g, o)
    org_idx = np.array([gp_to_org.get(i, -1) for i in nonzero])

    gp_ids = nodes.loc[nodes.node_type == "GeneProduct", "node_id"].tolist()
    org_ids = nodes.loc[nodes.node_type == "Organism", "node_id"].tolist()
    counts = pd.Series(org_idx).value_counts()
    top_taxids = [taxid_from_node_id(org_ids[o]) for o in counts.head(TOP_N_ORGANISMS).index if o >= 0]
    names_by_taxid = load_organism_names([t for t in top_taxids if t])
    org_name = {o: names_by_taxid.get(taxid_from_node_id(org_ids[o]), org_ids[o])
                for o in counts.head(TOP_N_ORGANISMS).index if o >= 0}

    labels, categories, colors = top_n_plus_other(gp_ids, org_idx, org_name)
    return X, labels, categories, colors, f"PlantCaduceus gene embeddings by organism ({len(nonzero):,} genes)"


def fig_organism_metabolite():
    data = torch.load(DATA_DIR / "heterodata.pt", weights_only=False)
    nodes = pd.read_csv(DATA_DIR / "nodes.tsv", sep="\t", low_memory=False)
    x = data["Metabolite"].x.numpy()
    nonzero = np.where(np.abs(x).sum(1) > 0)[0]
    X = x[nonzero]

    mo_key = ("Metabolite", "organism", "Organism")
    if mo_key in data.edge_types:
        mo_ei = data[mo_key].edge_index
        met_to_org = {}
        for m, o in zip(mo_ei[0].tolist(), mo_ei[1].tolist()):
            met_to_org.setdefault(m, o)
    else:
        met_to_org = {}
    org_idx = np.array([met_to_org.get(i, -1) for i in nonzero])

    met_ids = nodes.loc[nodes.node_type == "Metabolite", "node_id"].tolist()
    org_ids = nodes.loc[nodes.node_type == "Organism", "node_id"].tolist()
    counts = pd.Series(org_idx).value_counts()
    top_taxids = [taxid_from_node_id(org_ids[o]) for o in counts.head(TOP_N_ORGANISMS).index if o >= 0]
    names_by_taxid = load_organism_names([t for t in top_taxids if t])
    org_name = {o: names_by_taxid.get(taxid_from_node_id(org_ids[o]), org_ids[o])
                for o in counts.head(TOP_N_ORGANISMS).index if o >= 0}

    labels, categories, colors = top_n_plus_other(met_ids, org_idx, org_name)
    return X, labels, categories, colors, f"MAP4 metabolite embeddings by organism of origin ({len(nonzero):,} metabolites)"


def fig_ecclass_conversion():
    nodes = pd.read_csv(DATA_DIR / "nodes.tsv", sep="\t", low_memory=False)
    conv_emb = torch.load(DATA_DIR / "embeddings_conversion.pt", weights_only=False)
    inter = nodes[nodes.node_type == "Interaction"].reset_index(drop=True)
    conv = inter[inter["interaction_subtype"] == "Conversion"].reset_index(drop=True)

    ids, ecs = [], []
    for _, row in conv.iterrows():
        v = conv_emb.get(row["node_id"])
        if v is not None and v.abs().sum() > 0:
            ids.append(row["node_id"])
            ecs.append(row["ec_number"])

    X = np.stack([conv_emb[i].numpy() for i in ids])

    def top_ec(ec):
        try:
            return int(str(ec).split(".")[0])
        except (ValueError, AttributeError):
            return None

    labels = np.array([EC_NAMES.get(top_ec(e), "(no EC annotation)") for e in ecs])
    present = [c for c in EC_NAMES.values() if c in labels]
    categories = present + (["(no EC annotation)"] if "(no EC annotation)" in labels else [])
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(present)}
    colors["(no EC annotation)"] = OTHER_COLOR
    return X, labels, categories, colors, f"MAP4 DRFP-approx. Conversion fingerprints by EC top-level class ({len(ids):,} reactions)"


FIGURES = {
    "split_protein": ("embedding_split_protein", fig_split_protein),
    "organism_protein": ("embedding_organism_protein", fig_organism_protein),
    "organism_geneproduct": ("embedding_organism_geneproduct", fig_organism_geneproduct),
    "organism_metabolite": ("embedding_organism_metabolite", fig_organism_metabolite),
    "ecclass_conversion": ("embedding_ecclass_conversion", fig_ecclass_conversion),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(FIGURES), default=None)
    args = ap.parse_args()

    keys = args.only or list(FIGURES)
    for key in keys:
        out_name, loader = FIGURES[key]
        print(f"\n=== {key} ===")
        X, labels, categories, colors, subject = loader()
        print(f"  {X.shape[0]:,} points x {X.shape[1]}-dim, {len(categories)} categories")
        projections = reduce_all(X)
        make_figure(projections, labels, categories, colors, subject, out_name)


if __name__ == "__main__":
    main()
