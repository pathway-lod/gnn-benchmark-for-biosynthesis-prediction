# PlantMetWiki Full-Dataset Release

The complete PlantMetWiki plant-metabolism knowledge graph
(see https://doi.org/10.5281/zenodo.17967619), spanning 421 plant species,
with pre-computed node embeddings and three link-prediction split strategies.

This is the full-graph counterpart of the 2-species Learnathon subset
(https://doi.org/10.5281/zenodo.20736061) -- same task, same embedding
models, same file format, but the complete graph instead of an
A. thaliana / G. max slice.

## Changelog

- **v4** (2026-07-03): Learnathon baseline updated from S4 to **v6**.
  Data files are unchanged — the graph (`heterodata.pt`) and split (`splits_taxa.pt`)
  are identical to v2. Changes are in the learnathon training code and documentation:
  1. `dataset.py` adds `disjoint_train_ratio=0.2` (default): 80% of training positive
     edges enter the message-passing graph; 20% are supervision-only. Prevents the 1-hop
     shortcut where a model memorises training pairs from the MP graph.
  2. `dataset.py` removes `(Interaction, catalyzed_by, Protein)` from the MP graph by
     default (`keep_catalyzed_by=False`). This reverse edge created a 2-hop
     P → C → P shortcut that let the model memorise training pairs.
  3. `train.py` default negative sampling changed from 5+5 (random+CP) to 5+0:
     corrupt-protein negatives (`--neg-k-cp`) are no longer used by default. v6
     achieves CP-AUC=0.817 *without* CP training — and aggressively optimising
     CP-AUC via `--neg-k-cp` hurts P-H@50 (see §Baseline and §Metrics below).
  4. Learnathon metrics updated: **P-H@50** is the primary leaderboard metric,
     **CP-AUC/CP-AP** are the secondary reported metrics. Standard AUC/AP (random
     negatives) are omitted from leaderboard reporting as they are not informative
     (always ≥ 0.95 even for trivial models). `anyP-H@50` is an internal diagnostic
     not presented to participants.
  5. `is_about` edges (WikiPathways `wp:isAbout` provenance links) are confirmed
     load-bearing and must NOT be removed. The ablation (v7, 2026-07-03) that removed
     them caused catastrophic failure: test P-H@50 = 0.003 (below random baseline),
     AUC inverted to 0.188 — the GNN destroyed ESM-C embeddings rather than improving
     them. `heterodata.pt` contains these edges; the default training code keeps them.

- **v2**: two changes, bundled into this version since the first didn't ship
  before the second was found.
  1. `(Protein, catalyzes, Interaction)`'s destination is now always a
     `Conversion`-subtype node directly, not a `Catalysis`-subtype instance
     record one hop short of it -- collapsed by chaining
     `Protein -> Catalysis -> Conversion` (every `Catalysis` node has exactly
     one `Protein` and one `Conversion` neighbour, so this loses nothing but
     a handful of exact-duplicate pairs; see the source repo's
     `notebooks/01_explore_graph.ipynb` §8.1 for the cardinality
     confirmation). All three split files were rebuilt against this
     redefinition.
  2. Bugfix to `splits_taxa.pt`'s organism selection. The first rebuild
     picked held-out organisms by target-edge count alone, blind to
     embedding coverage -- since real Protein embedding coverage is almost
     entirely species-dependent (most non-model-organism identifiers don't
     resolve, see `notebooks/03_embeddings_fulldata.ipynb` §4.3), this landed
     val organisms at ~9% average Protein embedding coverage vs. train's
     ~64%, and inflated a HeteroConv-SAGE baseline's apparent val H@50 by
     ~4x relative to a properly-covered split (see that notebook's §7.7
     diagnosis and §7.8 rebuild). Now requires >=50% per-organism Protein
     embedding coverage for an organism to be eligible as a held-out
     val/test organism, giving more, better-covered held-out organisms (10
     val / 72 test, vs. 3 / 6 originally). `splits.pt`/`splits_pathway.pt`
     are unaffected by this fix (built independently of embedding coverage).
  Also bundles `training/fulldata_baseline_common.py` and
  `training/fulldata_baseline_train.py` (the HeteroConv-SAGE baseline that
  uses this target edge) as standalone reference code.
- **v1**: initial release, target edge was the `Catalysis`-subtype instance
  record directly.

## Files

### Graph

| File | Contents |
|---|---|
| `nodes.tsv` | 138,294 nodes -- columns `node_id`, `node_type`, `interaction_subtype` |
| `edges.tsv` | 338,935 edges -- columns `src`, `dst`, `rel`, `src_type`, `dst_type` |
| `heterodata.pt` | same graph as a PyTorch Geometric `HeteroData`, **including node features** (`x`) for Metabolite/Protein/GeneProduct from the embeddings below |
| `.provenance.json` | exact source data, Zenodo DOI, SHA-256, build parameters |

Node type counts:

| node_type | count |
|---|---|
| Interaction | 84,470 |
| DataNode | 0 |
| Metabolite | 17,476 |
| Protein | 13,682 |
| GeneProduct | 6,480 |
| Pathway | 2,478 |
| Organism | 424 |
| External | 0 |

### Node embeddings (already attached to `heterodata.pt[ntype].x` -- also included standalone for transparency)

Unresolved nodes get a zero feature vector, not exclusion from the graph --
they still receive GNN messages from neighbours after the first layer.

| File | Node type | Dim | Model | Coverage |
|---|---|---|---|---|
| `embeddings_metabolite.pt` | `Metabolite` | 1024 | MAP4 binary fingerprint (mhfp), 1024-dim | 4,199 / 17,476 (24.0%) |
| `embeddings_protein.pt` | `Protein` | 960 | ESM esmc_300m, 960-dim | 2,067 / 13,682 (15.1%) |
| `embeddings_gene.pt` | `GeneProduct` | 1024 | PlantCaduceus l32 fwd/rev avg, 1024-dim | 1,354 / 6,480 (20.9%) |

See `notebooks/03_embeddings_fulldata.ipynb` for the full resolution
diagnostics (namespace-by-namespace coverage, and known gaps -- e.g. the
`plantcyc`-namespace Protein/GeneProduct locus-extraction limitation, which
affects non-Arabidopsis species disproportionately).

### Train / val / test splits

All three target `(Protein, catalyzes, Interaction)` edges -- "this enzyme
catalyses this reaction" -- just with a different held-out strategy each.

| File | Strategy | train / val / test (positive edges) |
|---|---|---|
| `splits.pt` | Random 80/10/10 split on (Protein, catalyzes, Interaction) edges | 9,584 / 1,198 / 1,198 |
| `splits_taxa.pt` | Organism-held-out split -- val/test organisms never seen in train | 9,569 / 1,211 / 1,200 |
| `splits_pathway.pt` | Pathway-held-out split -- val/test pathway contexts never seen in train | 9,612 / 1,191 / 1,177 |

See `notebooks/02_build_splits.ipynb` for the full derivation, Tanimoto
diagnostics, and per-organism / per-pathway breakdowns.

## The task

Predict **`(Protein, catalyzes, Interaction)`** edges -- "this enzyme catalyses this reaction".
`Interaction` nodes on the right-hand side are `Conversion`-subtype
(`interaction_subtype == "Conversion"` in `nodes.tsv`), i.e. biochemical reactions;
`Protein` nodes are enzymes.

## Loading the graph

```python
import torch
data = torch.load("heterodata.pt", weights_only=False)   # already has .x for Metabolite/Protein/GeneProduct

# Pick a split strategy:
splits_random  = torch.load("splits.pt", weights_only=False)          # full HeteroData snapshots (also carry .x)
splits_taxa    = torch.load("splits_taxa.pt", weights_only=False)     # edge-index-only format
splits_pathway = torch.load("splits_pathway.pt", weights_only=False)  # edge-index-only format

TARGET_EDGE = ("Protein", "catalyzes", "Interaction")
```

`splits.pt`'s `train`/`val`/`test` are each a `HeteroData` with `edge_label_index`/
`edge_label` on `TARGET_EDGE` (1 = real edge, 0 = sampled negative) and already
carry node features. `splits_taxa.pt`/`splits_pathway.pt` instead store
`{train,val,test}_edge_index` tensors of `(protein_idx, interaction_idx)` pairs --
combine them with `data` (for features/message passing) directly, since they
reference the same node ordering as `nodes.tsv`/`heterodata.pt`.

## Why three split strategies?

- **`splits.pt` (random)**: i.i.d. baseline, fastest to validate a model end-to-end.
- **`splits_taxa.pt` (organism-held-out)**: tests generalisation across species --
  val/test edges come from organisms never seen in training. The realistic
  deployment scenario for a plant metabolomics tool. **Use this for the learnathon.**
- **`splits_pathway.pt` (pathway-held-out)**: tests generalisation across
  biological processes -- val/test edges come from pathway contexts never
  seen in training, complementary to the species-generalisation question above.

## Baseline (v6) and Metrics

**The learnathon baseline uses `splits_taxa.pt` with the following settings:**

```python
ctx = load_data(
    remove_is_part_of=True,      # strip Pathway co-membership edges (shortcut)
    disjoint_train_ratio=0.2,    # 80% of train positives in MP; 20% supervision-only
    keep_catalyzed_by=False,     # remove reverse edge from MP (closes 2-hop shortcut)
    embedded_only_ranking=True,  # rank only the ~8,445 embedded proteins
)
# Training: neg_k=5 (random Conversion negatives), neg_k_cp=0 (no CP negatives)
```

**Learnathon leaderboard metrics (report both):**

| Metric | Question | Random baseline | v6 baseline |
|--------|----------|-----------------|-------------|
| **P-H@50** *(primary)* | Is the true catalyst in the model's top-50 of 8,445 proteins? | 0.006 (50/8,445) | **val 0.124 · test 0.071** |
| **CP-AUC** *(secondary)* | Is the true (Protein, Reaction) scored above a random protein? | 0.50 | **val 0.817 · test 0.802** |

**Do not use as the primary metric:** standard AUC/AP (random negatives) always reaches
≥ 0.95 within a few epochs regardless of model quality — they measure "random proteins
score low," not enzyme specificity.

**CP-AUC gaming caveat:** CP-AUC can be directly optimised by training on corrupt-protein
negatives (`--neg-k-cp N`). Doing so aggressively collapses P-H@50 (seen in v5b: pure
CP-negative training drove P-H@50 to random while CP-AUC reached 0.609). Treat CP-AUC
as a diagnostic; optimise P-H@50.

**Key graph edges to understand before modifying the code:**

- `is_about` edges (`wp:isAbout`) are WikiPathways identity/provenance links — NOT
  similarity or homology. The `(Interaction, is_about, Interaction)` type (29,740 edges)
  connects the same reaction appearing in different organisms' pathway files. These are
  the only structural bridge enabling cross-organism knowledge transfer and must not be
  removed (ablation v7 confirmed: removal causes catastrophic failure).
- `is_part_of → Pathway` edges are removed by default — Pathway co-membership is a trivial
  shortcut that inflates AUC to ≥ 0.98 without biological learning.

## Regenerating

Graph: `scripts/fetch_zenodo.py` -> `scripts/unpack_ttl.py` -> `scripts/rdf_to_typed_tables.py`
-> `scripts/build_pyg.py` (see repo root `README.md` "Building the PyG").
Embeddings: `notebooks/03_embeddings_fulldata.ipynb` + `scripts/learnathon_embed_*.py`.
Splits: `notebooks/02_build_splits.ipynb`.
This README and `.provenance.json` are regenerated by
`scripts/generate_fulldata_archive_docs.py` -- re-run it after any of the above changes.
