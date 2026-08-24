# PlantMetBench

A graph machine learning benchmark for enzyme–reaction link prediction in plant metabolism.

**Task:** given a biochemical reaction from PlantMetWiki, identify which plant protein (enzyme) catalyses it.
The benchmark uses a heterogeneous knowledge graph of 424 plant species and evaluates models under a
cross-species taxa holdout — the model never sees the test organisms during training.

| Property | Value |
|---|---|
| Task | (Protein, catalyzes, Reaction) link prediction |
| Graph | PlantMetWiki, 424 plant species |
| Positive edges | 11,980 enzyme–reaction pairs |
| Ranking pool | 8,445 ESM-C embedded proteins |
| Split | Taxa holdout (11 val / 69 test organisms) |
| Primary metric | P-H@50 (protein ranking hits at 50) |
| Random P-H@50 | 50 / 8,445 ≈ 0.59% |
| Baseline P-H@50 | val 23.3% / test 18.9% (1-layer, no organisms) |

---

## Environments

Two conda environments are provided depending on what you want to do.

### For running GNN experiments (training, ablations, evaluation)

```bash
conda env create -f environment_training.yml
conda activate plantmetbench
```

This is the environment you need for **all paper results**.
Requires CUDA 12.1. For other CUDA versions, edit `pytorch-cuda=12.1` in the file.
For CPU-only, replace the two PyTorch lines with `pytorch::pytorch=2.4.*` and `pytorch::cpuonly`.

### For running data preparation notebooks

```bash
conda env create -f environment_data_prep.yml
conda activate plantmetbench-data
```

Use this to explore the graph, re-run EDA notebooks, or recompute the conversion/organism
embeddings from Zenodo inputs. See [data_preparation/README.md](data_preparation/README.md).

> **Note:** protein embeddings (ESM-C), gene embeddings (PlantCaduceus), metabolite fingerprints
> (MAP4), and EC one-hot embeddings are computed in the upstream pipeline and provided as
> pre-computed files on Zenodo. The data prep environment does **not** include those tools.

---

## Reproducing paper results — step by step

All commands below assume the **training environment** (`conda activate plantmetbench`)
and are run from the **`training/`** subdirectory unless stated otherwise.

### Step 0 — install and download data

```bash
conda env create -f environment_training.yml
conda activate plantmetbench
cd training/
python dataset.py       # downloads heterodata.pt + embeddings from Zenodo (~2 GB)
```

Data lands in `data/`. See [data/README.md](data/README.md) for manual download instructions.

---

### Step 1 — single baseline run (seed 42)

```bash
cd training/
python train.py
```

Expected output (after ~200 epochs, seed 42):

```
val  P-H@50 ≈ 0.175   CP-AUC ≈ 0.87   CP-AP ≈ 0.85
test P-H@50 ≈ 0.13    CP-AUC ≈ 0.84   CP-AP ≈ 0.80
```

Results are saved to `runs/baseline/report.json` (includes GPU timing + peak memory).
For multi-seed mean±std: val 23.3% ± 4.1%, test 18.9% ± 8.3% (5 seeds).

---

### Step 2 — multi-seed run (mean ± std)

```bash
cd training/
python train_seeds.py --run-name baseline
```

Runs seeds `42, 0, 1, 2, 3` sequentially (~5× training time).
Saves `runs/baseline/seeds_summary.json` **and** `results/baseline/seeds_summary.json`.

---

### Step 3 — shortcut ablations

```bash
cd training/
python ../scripts/run_ablations.py
```

Runs 5 experiments (baseline + 4 shortcuts re-enabled one at a time).
Saves `results/ablations.json`.

To run only one ablation or skip already-completed ones:

```bash
python ../scripts/run_ablations.py --only ec_features
python ../scripts/run_ablations.py --skip-existing
```

Ablations available: `baseline`, `keep_pathways`, `keep_catalyzed_by`, `no_disjoint`, `ec_features`.

---

### Step 4 — EC class distribution

```bash
cd training/
python ../scripts/extract_ec_distribution.py
```

Reads `data/embeddings_ec.pt` (downloaded in Step 0) and saves `results/ec_distribution.json`.

---

### Step 5 — GPU timing

Training time and peak GPU memory are logged automatically into every `report.json`.
After any run:

```bash
python -c "import json; r=json.load(open('runs/baseline/report.json')); print(r['compute_stats'])"
```

---

## All CLI flags for `train.py`

```bash
python train.py --help
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--hidden-dim` | 128 | GNN hidden dimension |
| `--num-layers` | 2 | Number of GNN layers |
| `--decoder` | dot | Predictor type (`dot` or `mlp`) |
| `--dropout` | 0.3 | Dropout between GNN layers |
| `--epochs` | 200 | Max training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--seed` | 42 | Random seed |
| `--neg-k` | 5 | Conversion-side negatives per positive |
| `--neg-k-cp` | 5 | Protein-side negatives per positive |
| `--early-stop-patience` | 50 | Epochs without val P-H@50 improvement |
| `--run-name` | baseline | Output directory under `runs/` |
| `--no-embeddings` | — | Replace pre-computed features with random (structural baseline) |
| `--ec-features` | — | Add EC one-hot to Interaction nodes (**ablation only**) |
| `--keep-pathways` | — | Keep `is_part_of` Pathway edges (**ablation only**) |
| `--keep-catalyzed-by` | — | Keep reverse catalysis edge (**ablation only**) |
| `--disjoint-train-ratio` | 0.2 | Fraction of train positives withheld from MP graph |
| `--organism-embeddings` | — | Path to `embeddings_organism.pt` for taxonomy-aware features |

---

## Dual-encoder baseline

A second standalone baseline lives in `dual_encoder/`. It requires only `torch`,
`numpy`, and `pandas` — no graph library — and can run on a laptop CPU.

```bash
cd dual_encoder/
pip install -r requirements.txt
python train.py          # 5 seeds, 200 epochs; writes results/results.json
```

See [dual_encoder/README.md](dual_encoder/README.md) for the method, hyperparameters,
and a discussion of the pool-size difference relative to the GNN baselines.

> **Comparability note.** The dual encoder evaluates over **2,232 distinct ESM vectors**
> (identical sequences deduplicated) rather than the full 8,445-node ranking pool used
> by the GNN and BLASTp baselines. The random baseline is therefore 50/2,232 = 2.24%
> rather than 50/8,445 = 0.59%. Direct comparison requires either re-evaluating the
> GNN on the deduplicated pool or re-evaluating the dual encoder on the full pool.

---

## Repository structure

```
plantmetbench/
├── environment_training.yml   ← install this for all GNN experiments
├── environment_data_prep.yml  ← install this for data prep notebooks only
│
├── data/                      ← downloaded automatically (gitignored)
│   └── README.md              ← Zenodo download instructions
│
├── data_preparation/          ← graph construction, EDA, and embedding notebooks
│   ├── README.md
│   ├── 01_explore_graph.ipynb
│   ├── 02_build_splits.ipynb
│   ├── 03_embeddings_fulldata.ipynb
│   ├── 04_link_prediction.ipynb
│   ├── 05_embedding_exploration.ipynb
│   └── scripts/
│       ├── compute_conversion_embeddings.py   ← recompute reaction fingerprints
│       ├── compute_ec_embeddings.py           ← recompute EC one-hot (upstream only)
│       ├── compute_organism_embeddings.py     ← recompute taxonomy MDS/multihot
│       └── package_embedding_inputs.py        ← bundle Zenodo inputs archive
│
├── training/                  ← GNN training code
│   ├── README.md
│   ├── dataset.py             ← data loading (auto-downloads from Zenodo)
│   ├── models.py              ← HeteroGNN + dot/MLP decoder
│   ├── metrics.py             ← P-H@K, CP-AUC/AP evaluation
│   ├── utils.py               ← negative sampling, seed setting
│   ├── report.py              ← JSON report + TeeLogger
│   ├── train.py               ← single-seed training
│   └── train_seeds.py         ← multi-seed run → mean ± std
│
├── dual_encoder/              ← contrastive retrieval baseline (no graph library needed)
│   ├── README.md              ← method, hyperparameters, results, comparability note
│   ├── requirements.txt       ← torch, numpy, pandas only
│   ├── model.py               ← MLPEncoder + DualEncoder
│   ├── losses.py              ← full-batch MLNCE loss
│   ├── data.py                ← benchmark loading, split remapping
│   ├── metrics.py             ← P-H@K, CP-AUC/AP
│   └── train.py               ← per-seed training loop + CLI
│
├── scripts/                   ← result-generation scripts
│   ├── run_ablations.py             ← → results/ablations.json
│   ├── extract_ec_distribution.py   ← → results/ec_distribution.json
│   └── eval_anyp.py                 ← any-catalyst P-H@K evaluation
│
└── results/                   ← machine-generated result files (tracked in git)
    ├── README.md              ← what each file is and how to regenerate it
    ├── ec_distribution.json
    ├── ablations.json
    ├── sage_1layer/seeds_summary.json
    ├── sage_2layer/seeds_summary.json
    ├── sage_3layer/seeds_summary.json
    ├── split_random/seeds_summary.json
    ├── split_pathway/seeds_summary.json
    ├── org_random/seeds_summary.json
    └── org_taxmds/seeds_summary.json
```

---

## Data

The dataset is derived from [PlantMetWiki](https://plantmetwiki.net), a curated knowledge graph
of plant metabolic pathways integrating WikiPathways-Plants, PlantCyc, and AraCyc.

**Zenodo:**
- Full dataset (graph + splits + embeddings): [10.5281/zenodo.20847651](https://doi.org/10.5281/zenodo.20847651)
- Embedding inputs (sequences, SMILES, EC numbers): [10.5281/zenodo.21237830](https://doi.org/10.5281/zenodo.21237830)

---

## How to cite

```bibtex
@inproceedings{plantmetbench2026,
  title     = {PlantMetBench: A Multimodal Knowledge Graph Benchmark for Plant Biosynthesis Prediction},
  author    = {Anonymous},
  booktitle = {NeurIPS 2026 Workshop},
  year      = {2026},
}
```
