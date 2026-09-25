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
embeddings from Zenodo inputs. See [notebooks/README.md](notebooks/README.md) and [data_preparation/README.md](data_preparation/README.md).

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

The script prints P-H@1/5/10/50, CP-AUC and CP-AP for the validation and test splits
at the best epoch, and saves them to `runs/baseline/.../report_*.json` (which also
includes GPU timing and peak memory). The reported numbers are in the paper.

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
| `--epochs` | 650 | Max training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--seed` | 42 | Random seed |
| `--neg-k` | 5 | Conversion-side negatives per positive |
| `--neg-k-cp` | 5 | Protein-side negatives per positive |
| `--early-stop-patience` | 300 | Epochs without val P-H@50 improvement |
| `--run-name` | baseline | Output directory under `runs/` |
| `--no-embeddings` | — | Replace pre-computed features with random (structural baseline) |
| `--ec-features` | — | Add EC one-hot to Interaction nodes (**ablation only**) |
| `--keep-pathways` | — | Keep `is_part_of` Pathway edges (**ablation only**) |
| `--keep-catalyzed-by` | — | Keep reverse catalysis edge (**ablation only**) |
| `--disjoint-train-ratio` | 0.2 | Fraction of train positives withheld from MP graph |
| `--organism-embeddings` | — | Path to `embeddings_organism.pt` for taxonomy-aware features |

---

## Dual-encoder baseline

A second standalone baseline lives in `baselines/dual_encoder/`. It requires only `torch`,
`numpy`, and `pandas` — no graph library — and can run on a laptop CPU.

```bash
cd baselines/dual_encoder/
pip install -r requirements.txt
python train.py          # 5 seeds, 200 epochs; writes results/results.json
```

See [baselines/dual_encoder/README.md](baselines/dual_encoder/README.md) for the method, hyperparameters,
and a discussion of the pool-size difference relative to the GNN baselines.

> **Pool-size comparability.** The dual encoder's default evaluation uses **2,232
> distinct ESM-C vectors** (identical protein sequences collapsed); the full pool has
> 8,445 proteins. Use `--keep_duplicates` to evaluate on the same 8,445-protein pool
> as the GNN and BLASTp baselines. Results on both pool sizes are in the paper appendix.

To evaluate on the 8,445-protein pool:

```bash
cd baselines/dual_encoder/
python train.py --seeds 42 0 1 2 3 --model_selection P-H@50 --num_layers 2 \
  --keep_duplicates \
  --checkpoint_dir checkpoints_fullpool \
  --output results/taxa_L2_ph50_fullpool.json
```

Use `--num_layers` 1 or 3 for the other depths. Checkpoints and
results for the default (2,232-pool) runs shown in
[baselines/dual_encoder/README.md](baselines/dual_encoder/README.md) omit `--keep_duplicates`.

---

## BLASTp baseline

A sequence-similarity baseline: for each evaluation reaction, every candidate
protein is scored by its best BLASTp bitscore against any training-set
catalyst of that reaction, then ranked. Method follows ReactZyme (Hua et al.,
2024), adapted for the reverse (reaction → protein) direction.

BLAST hits are precomputed and shipped in `baselines/blast/blast_results.tsv`, so
reproducing the reported numbers needs only the standard data release —
no local BLAST+ installation required. Complete
[Step 0](#step-0--install-and-download-data) first (environment + data
download), then from the repo root:

```bash
python baselines/blast/blast_evaluate.py                # test split, 8,445-protein pool
python baselines/blast/blast_evaluate.py --split val     # validation split
python baselines/blast/blast_evaluate.py --dedup-pool    # 2,232 deduplicated pool
```

Regenerating `blast_results.tsv` from raw sequences requires a local
`blastp`/`makeblastdb` install and is outside the scope of this script.

---

## Commands at a glance

`make help` lists the same targets. Each one is a thin wrapper, so `make -n <target>` shows the exact command.

| What | Command | Output |
|---|---|---|
| Download the dataset | `make data` | `data/` |
| One baseline run (seed 42) | `make baseline` | `runs/baseline/` |
| Baseline, 5 seeds | `make seeds` | `runs/baseline/`, `results/baseline/` |
| Another model, 5 seeds | `make model GNN=residual_jumping_sage LAYERS=2 RUN=resjump` | `runs/resjump/`, `results/resjump/` |
| Shortcut ablations | `make ablations` | `results/ablations.json` |
| EC class distribution | `make ec-distribution` | `results/ec_distribution.json` |
| Data-scaling ablation | `make train-frac` | `runs/frac-ablation/` |
| *A. thaliana* reaction holdout | `make ath-splits`, then `make ath` | `runs/ath_baseline/`, `results/ath_baseline/` |
| Dual encoder (2,232-protein pool) | `make dual-encoder` | `baselines/dual_encoder/results/` |
| Dual encoder (8,445-protein pool) | `make dual-encoder-fullpool` | `baselines/dual_encoder/results/` |
| Dual encoder L1-L3, deduplicated pool | `make dual-encoder-dedup` | `results/dual_encoder/` |
| BLASTp baseline | `make blast` (`make blast-dedup` for the 2,232 pool) | printed |
| Appendix tables from `results/` | `make tables` | printed |

Model names for `GNN=` are `sage`, `residual_sage`, `residual_jumping_sage` (HeteroSAGE+Res&Jump), `hgt`, `gat`, `rgcn` and `mix`.

---

## Repository structure

```
plantmetbench/
├── Makefile                   ← entry points for the experiments (`make help`)
├── environment_training.yml   ← install this for all GNN experiments
├── environment_data_prep.yml  ← install this for data prep notebooks only
│
├── data/                      ← downloaded automatically (gitignored)
│   └── README.md              ← Zenodo download instructions
│
├── notebooks/                 ← demonstration notebooks: graph, splits, embeddings, baseline, projections
│   ├── README.md
│   ├── 01_explore_graph.ipynb
│   ├── 02_build_splits.ipynb
│   ├── 03_embeddings_fulldata.ipynb
│   ├── 04_link_prediction.ipynb
│   └── 05_embedding_exploration.ipynb
│
├── data_preparation/          ← scripts that recompute embedding files from raw inputs
│   ├── README.md
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
│   ├── train_seeds.py         ← multi-seed run → mean ± std
│   ├── train_frac_ablation.py ← data-scaling ablation (--train-frac)
│   ├── plot_hidden_dim_ablation.py
│   └── plot_model_scatter.py
│
├── baselines/                 ← non-graph baselines
│   ├── dual_encoder/          ← contrastive retrieval baseline (no graph library needed)
│   │   ├── README.md          ← method, hyperparameters, comparability note
│   │   ├── requirements.txt   ← torch, numpy, pandas only
│   │   ├── model.py           ← MLPEncoder + DualEncoder
│   │   ├── losses.py          ← full-batch MLNCE loss
│   │   ├── data.py            ← benchmark loading, split remapping
│   │   ├── metrics.py         ← P-H@K, CP-AUC/AP
│   │   ├── eval_anyp.py       ← any-catalyst P-H@K evaluation
│   │   ├── run_dedup.sh       ← L1-L3 on the 2,232-protein deduplicated pool
│   │   └── train.py           ← per-seed training loop + CLI
│   └── blast/                 ← BLASTp baseline (self-contained)
│       ├── blast_evaluate.py  ← evaluates the precomputed hits on the benchmark splits
│       └── blast_results.tsv  ← precomputed hits (query, subject, bitscore); tracked in git
│
├── scripts/                   ← analysis, result-generation and plotting scripts
│   ├── run_ablations.py             ← shortcut ablations → results/ablations.json
│   ├── run_ath_baseline.sh          ← A. thaliana reaction-holdout runs
│   ├── extract_ec_distribution.py   ← → results/ec_distribution.json
│   ├── eval_anyp.py                 ← any-catalyst P-H@K evaluation of GNN checkpoints
│   ├── eval_pathway_reconstruction.py ← pathway reconstruction from ranked proteins
│   ├── eval_rhea_validation.py      ← Rhea cross-validation of high-ranked unlabelled proteins
│   ├── check_rhea_coverage.py       ← Rhea coverage of the benchmark reactions
│   ├── plot_coverage_figure.py      ← embedding-coverage figure
│   └── render_results_tables.py     ← prints the appendix tables from results/
│
├── figures/                   ← figures used in the paper and notebooks (PNG)
│
└── results/                   ← generated result files (not tracked; see results/README.md)
    └── README.md              ← what each file is and how to regenerate it

---

## Data

The dataset is derived from PlantMetWiki (2026,
[doi:10.64898/2026.07.22.733699](https://doi.org/10.64898/2026.07.22.733699)), a FAIR knowledge graph
of plant metabolic pathways built from PlantCyc, i.e. the Plant Metabolic Network
(Hawkins et al., 2025, *Nucleic Acids Research* 53(D1):D1606–D1613,
[doi:10.1093/nar/gkae991](https://doi.org/10.1093/nar/gkae991)).

**Zenodo:**
- Full dataset (graph + splits + embeddings): [10.5281/zenodo.20847651](https://doi.org/10.5281/zenodo.20847651)
- Embedding inputs (sequences, SMILES, EC numbers): [10.5281/zenodo.21237830](https://doi.org/10.5281/zenodo.21237830)

---

## Licence

**Code:** MIT (see [LICENSE](LICENSE)).

**Data:** the dataset is derived from PlantCyc, part of the Plant Metabolic Network (PMN), and is distributed under the terms of the *General Terms and Conditions of Open Database License for the Plant Metabolic Network Databases* (<https://plantcyc.org/webform/license-agreement/>). That licence permits royalty-free use, modification and redistribution for any purpose, provided that modified copies (i) identify the database they derive from, (ii) include its copyright notices and author lists, and (iii) summarise the modifications. The dataset archive contains the full licence text, the source attribution and the list of modifications in its `LICENSE.md`.
