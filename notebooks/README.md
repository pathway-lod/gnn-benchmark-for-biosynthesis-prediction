# Notebooks

Demonstration notebooks for the PlantMetBench data and code. They show how to inspect the
graph, the splits and the embeddings, and how the baseline is trained; they are not the
reported experiments (see the root README and `training/`).

## Notebooks

Run the notebooks in order. They locate the repo root and the `data/` folder themselves, so they can be started from `notebooks/` or from the repo root:

| Notebook | Description |
|----------|-------------|
| [01_explore_graph.ipynb](01_explore_graph.ipynb) | Load and explore the raw PlantMetWiki RDF graph: node/edge type distributions, organism coverage, pathway statistics |
| [02_build_splits.ipynb](02_build_splits.ipynb) | Construct the random and pathway-holdout splits (the taxa holdout is built in notebook 03) |
| [03_embeddings_fulldata.ipynb](03_embeddings_fulldata.ipynb) | Inspect and validate the pre-computed node embeddings (ESM-C proteins, MAP4 metabolites, PlantCaduceus genes): coverage by node type, namespace and organism, and the embedding-aware taxa split |
| [04_link_prediction.ipynb](04_link_prediction.ipynb) | Demonstration of the baseline: load the data with `training/dataset.py`, the shortcut controls, and training and evaluating a HeteroSAGE model as in `training/train.py` (not the reported results) |
| [05_embedding_exploration.ipynb](05_embedding_exploration.ipynb) | Visualise conversion (reaction) fingerprints: PCA / t-SNE / UMAP projections coloured by EC class |

### Running the notebooks

```bash
conda activate plantmetbench
cd notebooks/
jupyter lab
```
