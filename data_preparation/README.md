# Data Preparation

This folder contains notebooks and scripts for building and exploring the
PlantMetBench dataset from raw PlantMetWiki RDF data.

> **Note for reproducers**: The fully processed dataset (graph, splits, and
> embeddings) is available on Zenodo — see [data/README.md](../data/README.md).
> You only need to run these notebooks if you want to reproduce the data
> construction pipeline from scratch.

## Notebooks

Run the notebooks in order from the repo root (or adjust paths as needed):

| Notebook | Description |
|----------|-------------|
| [01_explore_graph.ipynb](01_explore_graph.ipynb) | Load and explore the raw PlantMetWiki RDF graph: node/edge type distributions, organism coverage, pathway statistics |
| [02_build_splits.ipynb](02_build_splits.ipynb) | Construct the taxa-holdout split: assign organisms to train/val/test, export `splits_taxa.pt` |
| [03_embeddings_fulldata.ipynb](03_embeddings_fulldata.ipynb) | Inspect and validate pre-computed node embeddings: ESM-C proteins, MAP4 metabolites, PlantCaduceus genes |
| [04_link_prediction.ipynb](04_link_prediction.ipynb) | End-to-end link prediction exploration: sanity checks on the P-H@K metric, shortcut analysis |
| [05_embedding_exploration.ipynb](05_embedding_exploration.ipynb) | Visualise conversion (reaction) fingerprints: PCA / t-SNE / UMAP projections coloured by EC class |

### Running the notebooks

```bash
conda activate plantmetbench
cd data_preparation/
jupyter lab
```

> **Path note**: notebooks were originally developed from the project root and
> reference paths like `../data/`. You may need to adjust these if running from
> inside `data_preparation/`.

## Embedding computation scripts

The `scripts/` subfolder contains scripts that produce the node embedding files
from raw inputs. These require external API access (NCBI, UniProt, PubChem) and
can take several hours to run. Use the Zenodo embedding inputs archive to skip them.

| Script | Input | Output | Notes |
|--------|-------|--------|-------|
| `compute_conversion_embeddings.py` | `data/` (metabolite SMILES from graph) | `embeddings_conversion.pt` | MAP4 DRFP-approx, 3072-dim |
| `compute_ec_embeddings.py` | `reactions.ttl` (RDF) | `embeddings_ec.pt` | 237-dim EC one-hot; adds `ec_number` column to `nodes.tsv` |
| `compute_organism_embeddings.py` | NCBI taxonomy (online) | `embeddings_organism.pt` | MDS 64-dim + multihot 702-dim from NCBI lineage |
| `package_embedding_inputs.py` | Various interim files | `embedding_inputs.zip` | Bundles resolved sequences/SMILES for sharing |

### Running the scripts

```bash
conda activate plantmetbench
# From repo root:
python data_preparation/scripts/compute_conversion_embeddings.py --data-dir data/
python data_preparation/scripts/compute_ec_embeddings.py --data-dir data/
python data_preparation/scripts/compute_organism_embeddings.py --data-dir data/
```

> The protein, gene, and metabolite embedding scripts (ESM-C, PlantCaduceus, MAP4)
> are not included here as they require GPU infrastructure and model downloads.
> See the Zenodo embedding inputs archive for the resolved input sequences/SMILES.
