"""Loading of the PlantMetBench reaction to protein retrieval task.

Inputs consumed from the benchmark release:

===========================  ============================================
``nodes.tsv``                node table; row order defines split indices
``embeddings_conversion.pt`` DRFP reaction fingerprints, 3072-d
``embeddings_protein.pt``    ESM protein embeddings, 960-d
``splits_taxa.pt``           taxa-holdout ``(Protein, catalyzes, Interaction)`` edges
===========================  ============================================

Split edge indices address the *global* node table, so they are remapped onto the
compact reaction/protein tables built here. Pairs whose reaction or protein lacks an
embedding are dropped, and the drop counts are reported.

The 8,445 embedded proteins share only 2,232 distinct ESM vectors: identical sequences
recur across accessions and organisms. An encoder reading nothing but the ESM vector
cannot separate the members of such a group, so by default the pool is collapsed to its
distinct vectors and a retrieval hit means the true protein's group was retrieved.
Keeping the duplicates instead makes P-H@1 structurally unreachable and inflates the
weight of over-represented sequences in the training partition function.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

DRFP_DIM = 3072
PROT_DIM = 960

SPLIT_KEYS = {"train": "train_edge_index", "val": "val_edge_index", "test": "test_edge_index"}


@dataclass(frozen=True)
class RetrievalData:
    """Embedding tables plus splits expressed as index pairs into those tables.

    Attributes:
        reaction_ids: Reaction URIs, aligned with ``reaction_x`` rows.
        reaction_x: DRFP fingerprints ``[R, 3072]``.
        protein_ids: Representative protein URI per candidate, aligned with
            ``protein_x`` rows. These rows are the retrieval pool.
        protein_x: ESM embeddings ``[P, 960]``.
        protein_members: All protein URIs behind each candidate. Singleton lists
            unless the pool was deduplicated.
        splits: ``{"train"|"val"|"test": LongTensor[2, N]}`` where row 0 holds
            reaction indices and row 1 holds protein-candidate indices.
    """

    reaction_ids: list[str]
    reaction_x: torch.Tensor
    protein_ids: list[str]
    protein_x: torch.Tensor
    protein_members: list[list[str]]
    splits: dict[str, torch.Tensor]

    @property
    def num_reactions(self) -> int:
        return len(self.reaction_ids)

    @property
    def pool_size(self) -> int:
        return len(self.protein_ids)


def load_plantmet(
    data_dir: str | Path,
    split_file: str = "splits_taxa.pt",
    deduplicate: bool = True,
    species_pool: bool = False,
    protein_embeddings_path: str | Path | None = None,
    verbose: bool = True,
) -> RetrievalData:
    """Build a :class:`RetrievalData` from a PlantMetBench release directory.

    Args:
        data_dir: Directory holding the benchmark release files.
        split_file: Which split definition to load, e.g. ``splits_taxa.pt``.
        deduplicate: Collapse proteins sharing an identical ESM vector into one
            retrieval candidate (see the module docstring).
        species_pool: Restrict the retrieval pool to ``split_file``'s
            ``ath_protein_idxs`` (only meaningful for ``splits_ath_pathway.pt``,
            mirroring ``dataset.load_data(..., species_pool=True)`` in the main
            GNN pipeline). Raises if the split file has no such field.
        protein_embeddings_path: Path to an alternate protein embeddings .pt
            file ({node_id: tensor}, any dimension). Defaults to
            data_dir/embeddings_protein.pt when not given.
        verbose: Print pool sizes and per-split drop counts.
    """
    data_dir = Path(data_dir)
    _log = print if verbose else (lambda *a, **k: None)
    _log(f"Loading PlantMetBench from {data_dir}")

    nodes = pd.read_csv(data_dir / "nodes.tsv", sep="\t", low_memory=False)
    node_proteins = nodes.loc[nodes["node_type"] == "Protein", "node_id"].tolist()
    node_reactions = nodes.loc[nodes["node_type"] == "Interaction", "node_id"].tolist()

    splits_raw = torch.load(data_dir / split_file, map_location="cpu", weights_only=False)

    # Candidate proteins: everything carrying an ESM embedding, in node-table order.
    prot_emb_path = Path(protein_embeddings_path) if protein_embeddings_path else data_dir / "embeddings_protein.pt"
    protein_emb = torch.load(prot_emb_path, map_location="cpu", weights_only=False)
    if species_pool:
        if "ath_protein_idxs" not in splits_raw:
            raise ValueError(
                f"species_pool=True but {split_file} has no ath_protein_idxs field"
            )
        allowed = set(splits_raw["ath_protein_idxs"].tolist())
        embedded_ids = [p for i, p in enumerate(node_proteins)
                         if p in protein_emb and i in allowed]
        _log(f"  species pool : restricted to {len(allowed):,} A. thaliana protein indices")
    else:
        embedded_ids = [p for p in node_proteins if p in protein_emb]
    embedded_x = torch.stack([protein_emb[p].float() for p in embedded_ids])
    _log(f"  embedded prot: {len(embedded_ids):,} / {len(node_proteins):,} nodes have ESM embeddings")

    protein_ids, protein_x, protein_members, protein_index = _build_pool(
        embedded_ids, embedded_x, deduplicate
    )
    suffix = " (distinct ESM vectors)" if deduplicate else ""
    _log(f"  pool size    : {len(protein_ids):,} retrieval candidates{suffix}")

    reaction_emb = torch.load(data_dir / "embeddings_conversion.pt", map_location="cpu", weights_only=False)
    _log(f"  DRFP table   : {len(reaction_emb):,} reactions")

    if protein_embeddings_path is None:
        _check_dim(protein_x.shape[1], PROT_DIM, "protein")
    _check_dim(next(iter(reaction_emb.values())).shape[-1], DRFP_DIM, "reaction")

    # Index reactions lazily so the reaction table holds only what the splits use.
    reaction_ids: list[str] = []
    reaction_index: dict[str, int] = {}
    splits: dict[str, torch.Tensor] = {}

    for name, key in SPLIT_KEYS.items():
        edge_index = splits_raw[key]  # [2, N] with row 0 = protein, row 1 = interaction
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        dropped_protein = dropped_reaction = merged = 0

        for protein_node, reaction_node in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            pid = node_proteins[protein_node]
            rid = node_reactions[reaction_node]
            if pid not in protein_index:
                dropped_protein += 1
                continue
            if rid not in reaction_emb:
                dropped_reaction += 1
                continue
            if rid not in reaction_index:
                reaction_index[rid] = len(reaction_ids)
                reaction_ids.append(rid)

            # Two proteins collapsed into one candidate can yield the same pair twice.
            pair = (reaction_index[rid], protein_index[pid])
            if pair in seen:
                merged += 1
                continue
            seen.add(pair)
            pairs.append(pair)

        splits[name] = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        _log(
            f"  {name:<5} split : {len(pairs):,} pairs "
            f"(dropped {dropped_protein:,} no-embedding proteins, "
            f"{dropped_reaction:,} no-DRFP reactions, merged {merged:,} duplicates)"
        )

    reaction_x = torch.stack([reaction_emb[rid].float() for rid in reaction_ids])
    return RetrievalData(reaction_ids, reaction_x, protein_ids, protein_x, protein_members, splits)


def _build_pool(
    ids: list[str], x: torch.Tensor, deduplicate: bool
) -> tuple[list[str], torch.Tensor, list[list[str]], dict[str, int]]:
    """Turn embedded proteins into retrieval candidates, optionally collapsing duplicates.

    Returns the representative id per candidate, the candidate embedding matrix, the
    protein ids behind each candidate, and a lookup from protein id to candidate row.
    """
    if not deduplicate:
        return ids, x, [[pid] for pid in ids], {pid: i for i, pid in enumerate(ids)}

    unique_x, inverse = torch.unique(x, dim=0, return_inverse=True)
    members: list[list[str]] = [[] for _ in range(unique_x.shape[0])]
    for pid, candidate in zip(ids, inverse.tolist()):
        members[candidate].append(pid)
    representatives = [group[0] for group in members]
    index = {pid: candidate for candidate, group in enumerate(members) for pid in group}
    return representatives, unique_x, members, index


def _check_dim(observed: int, expected: int, what: str) -> None:
    if observed != expected:
        raise ValueError(f"expected {expected}-d {what} embeddings, found {observed}-d")
