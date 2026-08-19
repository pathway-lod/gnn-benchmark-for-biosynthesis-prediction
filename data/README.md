# Data

Data files are **not stored in this repository** — they are downloaded automatically
on first run, or can be obtained manually from Zenodo.

## Automatic download

```bash
cd ../training
python train.py          # downloads data/ automatically on first run
```

## Manual download

| Record | Contents | DOI |
|--------|----------|-----|
| PlantMetBench full dataset | `heterodata.pt`, `splits_taxa.pt`, `nodes.tsv`, `edges.tsv`, embeddings | [10.5281/zenodo.20847651](https://doi.org/10.5281/zenodo.20847651) |
| Embedding inputs | Protein sequences, gene sequences, metabolite SMILES/InChIKeys, conversion EC numbers, organism NCBI IDs | [10.5281/zenodo.21237830](https://doi.org/10.5281/zenodo.21237830) |

After downloading, place all files directly in this `data/` directory.
The training code expects:
- `heterodata.pt`      — PyTorch Geometric HeteroData object
- `splits_taxa.pt`     — taxa-holdout split indices
- `nodes.tsv`          — node metadata (IDs, types, subtypes)
- `edges.tsv`          — edge list (needed for pathway pool construction)
- `embeddings_protein.pt`    — ESM-C 960-dim protein embeddings (optional, loaded if present)
- `embeddings_conversion.pt` — MAP4 3072-dim reaction fingerprints (optional, loaded if present)
- `embeddings_organism.pt`   — taxonomy MDS / multihot (optional, for `--organism-embeddings`)
