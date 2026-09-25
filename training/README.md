# Training

Heterogeneous GNN for enzyme–reaction link prediction on PlantMetBench.

**Task**: given a biochemical reaction (Conversion node), rank the 8,445 embedded
plant proteins to find the true enzyme catalyst.

**Split**: taxa holdout — val/test reactions belong to organisms not seen during training.

## Quick start

```bash
conda activate plantmetbench
cd training/

# Baseline (single seed, reproduces paper Table X)
python train.py

# 5-seed run — saves mean ± std to runs/baseline/seeds_summary.json
python train_seeds.py

# Structural baseline (random node features, no ESM/MAP4)
python train.py --no-embeddings --run-name structural_baseline

# Larger model
python train.py --hidden-dim 256 --num-layers 3 --run-name larger_model
```

Results are saved to `../runs/<run-name>/`:
- `train.log`       — full stdout log
- `history.json`    — per-epoch metrics
- `report.json`     — final results JSON (hparams + dataset stats + metrics)
- `<name>_best.pt`  — best checkpoint (by val P-H@50)

For multi-seed runs, `seeds_summary.json` is saved to `../runs/<run-name>/`
with `mean`, `std`, and per-seed values for every metric.

## Baseline configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | HeteroSAGE | SAGEConv per edge type, `aggr=sum` |
| Hidden dim | 128 | |
| Layers | 2 | min 2 for 2-hop neighbourhood |
| Decoder | dot product | simple, parameter-free |
| Dropout | 0.3 | between layers only |
| Epochs | 200 | early stop patience 50 |
| Optimizer | AdamW | lr=1e-4, weight_decay=1e-5 |
| Negatives | 5 × (P, C_rand) + 5 × (P_rand, C) | both directions |
| Protein features | ESM-C 960-dim | pre-computed |
| Reaction features | MAP4 DRFP-approx 3072-dim | pre-computed |
| Pathway edges | removed | prevents co-membership shortcut |
| Reverse `catalyzed_by` | removed | prevents 2-hop shortcut |
| Disjoint train ratio | 0.2 | prevents 1-hop shortcut |

## Expected results (seed=42)

| Split | P-H@50 | CP-AUC | CP-AP |
|-------|--------|--------|-------|
| Val   | 11.5%  | 0.906  | 0.858 |
| Test  | 6.0%   | 0.891  | 0.817 |

Random baseline: P-H@50 = 50/8445 ≈ **0.59%**

## Metrics

**P-H@50** (primary): fraction of val/test reactions where the true catalyst is
ranked in the top 50 out of 8,445 embedded proteins. This is the research-question
metric — it directly measures protein-ranking quality.

**CP-AUC / CP-AP** (secondary): AUC and average precision for distinguishing
the true (Protein, Reaction) pair from a random-protein pair for the same reaction.

> **Do not use the random-negative AUC** (`evaluate_random_neg_auc`) as a training
> signal — it inflates to ~0.98 due to Pathway co-membership even for random
> embeddings. It is logged only for historical comparability.

## Shortcut ablations

| Ablation | Effect |
|----------|--------|
| `--keep-catalyzed-by` (edit dataset.py) | 2-hop P→C→P shortcut; inflates val AUC |
| `--ec-features` | EC one-hot gives reaction-class identity → ~80% val→test gap |
| `disjoint_train_ratio=0` (edit dataset.py) | 1-hop shortcut; model memorises training pairs |
| Without `remove_is_part_of` (edit dataset.py) | Pathway membership inflates AUC to ~0.98 |

## Files

| File | Description |
|------|-------------|
| `dataset.py` | Download + load the graph; `load_data()` returns a `GraphContext` |
| `models.py` | `HeteroGNN` (SAGEConv), `DotPredictor`, `MLPPredictor` — edit to change architecture |
| `metrics.py` | `protein_hits_at_k`, `evaluate_cp_auc`, `evaluate_random_neg_auc` |
| `utils.py` | Seed setting, negative sampling, pathway pool construction |
| `report.py` | `TeeLogger` (stdout → file), `save_report` (JSON) |
| `train.py` | Single-seed training loop |
| `train_seeds.py` | Multi-seed wrapper: runs 5 seeds, saves `seeds_summary.json` |
| `train_frac_ablation.py` | Data-scaling ablation: trains on increasing fractions of the training positives (`--train-frac`) and plots val/test P-H@50 and CP-AUC |
| `plot_hidden_dim_ablation.py` | Bar chart of test P-H@K across hidden dimensions for HeteroSAGE+Res&Jump |
| `plot_model_scatter.py` | Scatter of test P-H@50 vs. best epoch, with marker size proportional to the parameter count |

## All CLI flags

```
python train.py --help
```

Key flags:
```
--hidden-dim INT        GNN embedding dimension (default 128)
--num-layers INT        Number of HeteroConv layers (default 2)
--decoder dot|mlp       Link decoder (default dot)
--dropout FLOAT         Dropout rate (default 0.3)
--epochs INT            Training epochs (default 200)
--seed INT              Random seed (default 42)
--no-embeddings         Structural baseline: random node features
--ec-features           Append EC one-hot (ablation only — causes val→test gap)
--remove-all-metabolites  Drop Metabolite nodes; reactions only from fingerprints
--organism-embeddings PATH  Replace random Organism features with taxonomy MDS
--run-name STR          Output directory name (runs/<name>/)
```
