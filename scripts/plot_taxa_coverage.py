#!/usr/bin/env python3
"""
Per-organism supervision-edge coverage in the taxa split: which organisms
contribute the most (Protein, catalyzes, Interaction) target edges, and how
that contribution is distributed across all organisms.

Panel A: top 25 organisms by target-edge count, labelled with resolved NCBI
scientific names (and common names where available) rather than raw
NCBITaxon_XXXXX identifiers.
Panel B: log-scale distribution of target-edge count across all organisms.

Organism names are resolved once via the NCBI Taxonomy efetch API and cached
locally (data/organism_names_cache.json) so re-runs don't require network
access.

Usage (from repo root):
    python scripts/plot_taxa_coverage.py
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)
NAME_CACHE = DATA_DIR / "organism_names_cache.json"

NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 50
REQUEST_DELAY = 0.34

C = dict(bar="#3B70A3", hist="#3B70A3", grid="#E3E8EE", text="#1A1E2A", sub="#5A6070",
         train="#3B70A3", val="#C2782E", test="#3D8A5C")


def taxid_from_node_id(node_id: str) -> str | None:
    m = re.search(r"NCBITaxon_(\d+)", node_id)
    return m.group(1) if m else None


def resolve_organism_names(taxids: list[str]) -> dict[str, str]:
    """Return {taxid: "Scientific name (common name)"} via NCBI efetch,
    using a local JSON cache so repeated runs need no network access."""
    cache: dict[str, str] = {}
    if NAME_CACHE.exists():
        cache = json.loads(NAME_CACHE.read_text())

    missing = [t for t in taxids if t not in cache]
    if missing:
        import xml.etree.ElementTree as ET
        print(f"  Resolving {len(missing)} organism name(s) via NCBI efetch …")
        for i in range(0, len(missing), BATCH_SIZE):
            batch = missing[i:i + BATCH_SIZE]
            r = requests.get(NCBI_EFETCH, params={
                "db": "taxonomy", "id": ",".join(batch), "retmode": "xml"}, timeout=60)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            for taxon in root.findall("Taxon"):
                tid = taxon.findtext("TaxId", "").strip()
                sci = taxon.findtext("ScientificName", "").strip()
                common = taxon.findtext("OtherNames/GenbankCommonName", "").strip()
                cache[tid] = f"{sci} ({common})" if common and common.lower() != sci.lower() else sci
            time.sleep(REQUEST_DELAY)
        NAME_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))

    return cache


def load_target_edge_counts() -> pd.DataFrame:
    data = torch.load(DATA_DIR / "heterodata.pt", weights_only=False)
    nodes = pd.read_csv(DATA_DIR / "nodes.tsv", sep="\t", low_memory=False)
    org_ids = nodes.loc[nodes.node_type == "Organism", "node_id"].tolist()

    cat_ei = data[("Protein", "catalyzes", "Interaction")].edge_index
    prot_org_ei = data[("Protein", "organism", "Organism")].edge_index
    prot_to_org: dict[int, int] = {}
    for p, o in zip(prot_org_ei[0].tolist(), prot_org_ei[1].tolist()):
        prot_to_org.setdefault(p, o)  # one organism per protein

    org_counts = pd.Series(0, index=range(len(org_ids)), dtype=int)
    counts: dict[int, int] = {}
    for p in cat_ei[0].tolist():
        o = prot_to_org.get(p)
        if o is not None:
            counts[o] = counts.get(o, 0) + 1
    for o, c in counts.items():
        org_counts.loc[o] = c

    df = pd.DataFrame({"node_id": org_ids, "target_edges": org_counts.values})
    df["taxid"] = df["node_id"].map(taxid_from_node_id)

    splits = torch.load(DATA_DIR / "splits_taxa.pt", weights_only=False)
    val_orgs = set(splits["val_organisms"])
    test_orgs = set(splits["test_organisms"])
    df["split"] = "train"
    df.loc[df.index.isin(val_orgs), "split"] = "val"
    df.loc[df.index.isin(test_orgs), "split"] = "test"
    return df


def make_figure(df: pd.DataFrame, names: dict[str, str], top_n: int = 25):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 17,
        "axes.linewidth": 0.8,
        "xtick.major.size": 3.5, "xtick.major.width": 0.7,
        "ytick.major.size": 3.5, "ytick.major.width": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    top = df.sort_values("target_edges", ascending=False).head(top_n).iloc[::-1]
    labels = [names.get(t, nid) for t, nid in zip(top["taxid"], top["node_id"])]
    bar_colors = [C[s] for s in top["split"]]

    fig = plt.figure(figsize=(13.5, 8.4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1.0],
                            left=0.33, right=0.97, bottom=0.10, top=0.82, wspace=0.48)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # ── Panel A: top-N organisms by target-edge count, coloured by split ───────
    ax_a.barh(range(len(top)), top["target_edges"], color=bar_colors, alpha=0.85,
              edgecolor="white", linewidth=0.6, height=0.68)
    ax_a.set_yticks(range(len(top)))
    ax_a.set_yticklabels(labels, fontsize=15.5, style="italic")
    for tick, split in zip(ax_a.get_yticklabels(), top["split"]):
        tick.set_color(C[split])
    ax_a.set_xlabel("Target edges", fontsize=18)
    ax_a.set_title(r"$\bf{A}$   " + f"Target edges per organism (top {top_n})",
                    fontsize=19, color=C["text"], pad=12, loc="left")
    ax_a.tick_params(axis="x", labelsize=15.5)
    ax_a.xaxis.grid(True, linewidth=0.5, linestyle="--", color=C["grid"], zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.spines["left"].set_color("#D0D8E0")
    ax_a.spines["bottom"].set_color("#D0D8E0")
    ax_a.tick_params(colors=C["sub"], length=3.5)
    ax_a.set_ylim(-0.7, len(top) - 0.3)

    # ── Panel B: log-scale distribution across all organisms ───────────────────
    log_counts = np.log1p(df["target_edges"])
    ax_b.hist(log_counts, bins=30, color=C["hist"], alpha=0.75,
              edgecolor="white", linewidth=0.6)
    ax_b.set_xlabel("log(1 + target edges)", fontsize=18)
    ax_b.set_ylabel("Number of organisms", fontsize=18)
    ax_b.set_title(r"$\bf{B}$   " + "Organism size (log scale)",
                    fontsize=19, color=C["text"], pad=12, loc="left")
    ax_b.tick_params(labelsize=15.5)
    ax_b.yaxis.grid(True, linewidth=0.5, linestyle="--", color=C["grid"], zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.spines["left"].set_color("#D0D8E0")
    ax_b.spines["bottom"].set_color("#D0D8E0")
    ax_b.tick_params(colors=C["sub"], length=3.5)

    # ── Shared legend for split colour coding ───────────────────────────────────
    import matplotlib.patches as mpatches
    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())
    n_test = int((df["split"] == "test").sum())
    handles = [
        mpatches.Patch(color=C["train"], alpha=0.85, label=f"Training (n={n_train})"),
        mpatches.Patch(color=C["val"], alpha=0.85, label=f"Validation (n={n_val})"),
        mpatches.Patch(color=C["test"], alpha=0.85, label=f"Test (n={n_test})"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=15.5,
               frameon=True, framealpha=0.95, edgecolor="#D8DDE4",
               handlelength=1.2, handleheight=1.0, borderpad=0.6,
               columnspacing=1.8, bbox_to_anchor=(0.63, 1.0), borderaxespad=0.3)

    return fig


def main():
    print("Loading per-organism target-edge counts …")
    df = load_target_edge_counts()
    top_taxids = df.sort_values("target_edges", ascending=False).head(25)["taxid"].dropna().tolist()
    names = resolve_organism_names(top_taxids)

    print(f"  {len(df)} organisms total, {int((df['target_edges']>0).sum())} with >=1 target edge")
    print("  Top 5:")
    for _, row in df.sort_values("target_edges", ascending=False).head(5).iterrows():
        print(f"    {names.get(row['taxid'], row['node_id']):<40} {row['target_edges']:,} edges")

    fig = make_figure(df, names)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"taxa_coverage.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
