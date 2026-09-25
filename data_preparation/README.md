# Data Preparation

This folder contains the scripts that recompute the embedding files of the
PlantMetBench dataset from raw inputs. The notebooks are in [../notebooks/](../notebooks/).

> **Note for reproducers**: The fully processed dataset (graph, splits, and
> embeddings) is available on Zenodo — see [data/README.md](../data/README.md).
> You only need to run these scripts if you want to reproduce the data
> construction pipeline from scratch.

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
