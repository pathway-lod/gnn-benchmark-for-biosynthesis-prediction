# PlantMetBench

A graph machine learning benchmark for enzyme–reaction link prediction in plant metabolism.

Given a biochemical reaction from PlantMetWiki, the task is to identify which plant protein
(enzyme) catalyses it. The benchmark uses a heterogeneous knowledge graph of 424 plant species
and evaluates models under a cross-species taxa holdout.

## Overview

| Property | Value |
|----------|-------|
| Task | (Protein, catalyzes, Reaction) link prediction |
| Graph | PlantMetWiki, 424 plant species |
| Positive edges | 11,980 enzyme–reaction pairs |
| Ranking pool | 8,445 embedded proteins |
| Split | Taxa holdout (10 val / 69 test organisms) |
| Primary metric | P-H@50 (protein ranking hits at 50) |
| Random baseline | 50/8445 ≈ 0.59% |
| Baseline P-H@50 | val 11.5% / test 6.0% |

## Installation

```bash
conda env create -f environment.yml
conda activate plantmetbench
```

Requires CUDA 12.1. For other CUDA versions, edit `pytorch-cuda=12.1` in `environment.yml`.
For CPU-only, replace the two PyTorch lines with `pytorch::pytorch=2.4.*` and `pytorch::cpuonly`.

## Reproduce the baseline

```bash
# Single run (seed 42)
cd training/
python train.py

# 5-seed run — outputs mean ± std to runs/baseline/seeds_summary.json
python train_seeds.py
```

Data is downloaded automatically from Zenodo on first run (~2 GB).
See [data/README.md](data/README.md) for manual download instructions.

## Expected results

| Model | Split | P-H@50 | CP-AUC | CP-AP |
|-------|-------|--------|--------|-------|
| HeteroSAGE (seed 42) | Val  | 11.5% | 0.906 | 0.858 |
| HeteroSAGE (seed 42) | Test |  6.0% | 0.891 | 0.817 |

## Repository structure

```
plantmetbench/
├── environment.yml          # conda environment
├── data/                    # downloaded data (gitignored)
│   └── README.md            # Zenodo download instructions
├── data_preparation/        # graph construction and embedding notebooks
│   ├── README.md
│   ├── 01_explore_graph.ipynb
│   ├── 02_build_splits.ipynb
│   ├── 03_embeddings_fulldata.ipynb
│   ├── 04_link_prediction.ipynb
│   ├── 05_embedding_exploration.ipynb
│   └── scripts/             # embedding computation scripts
│       ├── compute_conversion_embeddings.py
│       ├── compute_ec_embeddings.py
│       ├── compute_organism_embeddings.py
│       └── package_embedding_inputs.py
└── training/                # GNN training code
    ├── README.md            # training guide and all CLI flags
    ├── dataset.py           # data loading (auto-downloads from Zenodo)
    ├── models.py            # HeteroGNN architecture (edit to change model)
    ├── metrics.py           # P-H@K, CP-AUC/AP evaluation
    ├── utils.py             # negative sampling, seed setting
    ├── report.py            # logging and JSON report utilities
    ├── train.py             # single-seed training
    └── train_seeds.py       # multi-seed training with mean ± std
```

## Data

The dataset is derived from [PlantMetWiki](https://plantmetwiki.net), a curated knowledge graph
of plant metabolic pathways. The graph contains proteins, metabolites, biochemical reactions
(Conversions), genes, pathways, and organisms connected by typed edges.

**Zenodo records:**
- Full dataset (graph + splits + embeddings): [10.5281/zenodo.20847651](https://doi.org/10.5281/zenodo.20847651)
- Embedding inputs (sequences, SMILES, EC numbers): [10.5281/zenodo.21237830](https://doi.org/10.5281/zenodo.21237830)

## Citation

```bibtex
@inproceedings{plantmetbench2025,
  title     = {PlantMetBench: A Heterogeneous Graph Benchmark for Plant Metabolic Enzyme Prediction},
  author    = {Anonymous},
  booktitle = {NeurIPS 2025 Workshop},
  year      = {2025},
}
```
