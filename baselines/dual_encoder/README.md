# Dual-encoder contrastive retrieval on PlantMetBench

A compact baseline for the PlantMetBench **reaction → protein** retrieval task: given a
metabolic reaction, rank every candidate protein in the pool by how likely it is to
catalyse that reaction.

Two MLP towers embed the two modalities into a shared, L2-normalised space — reactions
from DRFP reaction fingerprints (3072-d), proteins from ESM embeddings (960-d) — and are
trained with a full-batch maximum-likelihood NCE objective. At inference, retrieval is a
single matrix product: cosine similarity between the query reaction and the protein pool.

## Task and data

The benchmark release supplies everything needed; no graph library is required.

| File | Used for |
| --- | --- |
| `nodes.tsv` | node table; its row order defines the split indices |
| `embeddings_conversion.pt` | DRFP reaction fingerprints (3072-d) |
| `embeddings_protein.pt` | ESM protein embeddings (960-d) |
| `splits_taxa.pt` | taxa-holdout `(Protein, catalyzes, Interaction)` edges |

Splits are **taxa holdout**: validation and test organisms are disjoint from training
organisms, so a model cannot succeed by memorising species-specific patterns.

The retrieval pool is identical for training negatives and for evaluation. Positive pairs
whose reaction or protein lacks an embedding are dropped, and the counts are printed at
load time.

**Duplicate sequences.** 8,445 proteins carry an ESM embedding, but they share only
**2,232 distinct vectors** — identical sequences recur across accessions and organisms.
An encoder reading nothing but the ESM vector cannot separate members of such a group, so
the pool is collapsed to its distinct vectors by default and a hit means the true
protein's group was retrieved. Keeping the duplicates makes P-H@1 unreachable for *any*
model (a duplicate always ties with the positive, so the best attainable rank is 2) and
lets over-represented sequences dominate the training partition function. Pass
`--keep_duplicates` to score all 8,445 accessions instead.

## Method

**Encoders.** Each tower is `num_layers` blocks of `Linear → ReLU → LayerNorm → Dropout`
followed by a linear projection to `emb_dim` and L2 normalisation. The towers share no
weights — only the output space.

**Objective.** Full-batch MLNCE over the dense reaction × protein cosine-distance matrix:

```
L = β · mean_(i,j)∈P d_ij  +  log Σ_i Σ_j exp(−β · d_ij)
```

The first term pulls annotated pairs together, the log-partition term over the whole
matrix pushes everything else apart. Because positives are supplied as an index set
rather than a diagonal, a reaction with several catalysing proteins is handled natively —
no one-positive-per-row assumption, and no negative sampling: every epoch contrasts each
training reaction against the entire pool.

**Metrics.**

- **P-H@K** — fraction of positive pairs whose true protein lands in the top K of the
  full pool.
- **CP-AUC / CP-AP** — each positive scored against 50 proteins sampled without
  replacement from the pool. The negative-sampling seed is fixed independently of the
  training seed, so all runs are evaluated under one protocol.

Ties count against the model everywhere. With one positive per query, the closed forms
in `metrics.py` are exactly equal to scikit-learn's `roc_auc_score` and
`average_precision_score`, which is why that dependency is not needed.

## Usage

Requires only `torch`, `numpy` and `pandas` — no graph library. Results below were
produced with PyTorch 2.11 (CUDA 12.8) on Python 3.11.

```bash
pip install -r requirements.txt

python train.py                                     # 5 seeds (42, 0, 1, 2, 3), 200 epochs
python train.py --data_dir /path/to/plantmetbench   # release directory elsewhere
python train.py --seeds 42 --epochs 50              # quick single-seed check
```

By default `--data_dir` points at `../../data`. Each seed trains independently; the
checkpoint with the best validation P-H@50 is the one evaluated on test. Per-seed
metrics and the aggregate land in `results/results.json`, checkpoints in
`checkpoints/dual_encoder_seed<N>.pt`.

## Hyperparameters

| | |
| --- | --- |
| Hidden layers / width | 2 × 2048 |
| Embedding dimension | 512 |
| Dropout | 0.3 |
| Inverse temperature β | 10.0 |
| Optimiser | AdamW |
| Learning rate | 1 × 10⁻⁴, cosine annealed to 0 |
| Weight decay | 1 × 10⁻⁵ |
| Gradient clipping | 1.0 |
| Epochs | 200 (full-batch, one step per epoch) |
| Seeds | 42, 0, 1, 2, 3 |

## Results

Test set (930 pairs), taxa holdout, 2,232-candidate pool. Mean ± sample standard
deviation over seeds 42, 0, 1, 2, 3.

> **Not directly comparable to the GNN baselines in `results/`.** Those rank over all
> 8,445 protein nodes, where the random baseline is 50/8445 = 0.59%; here it is
> 50/2232 = 2.24%. Since higher-scoring enzymes occupy ~3.8 node slots each in the full
> pool, top-50 of 8,445 corresponds to roughly the top 13 distinct enzymes — so P-H@50
> below is a more permissive measurement, not a rescaling of the full-pool figure. A
> like-for-like comparison requires re-evaluating the GNN under the same group-collapsing
> rule, scoring each group by the max over its members.

The script reports P-H@1/5/10/50, CP-AUC and CP-AP as mean and standard deviation over
the five seeds, and the number of model parameters and the training time per epoch.
Epoch time measures the training step alone (forward, backward, optimiser), excluding
validation passes, so it does not depend on the `--eval_every` cadence. The reported
numbers are in the paper.

> Run `python train.py` to regenerate `results/results.json` and the per-seed
> checkpoints; they are gitignored and not shipped with the repository.

## Layout

```
model.py     MLPEncoder and DualEncoder
losses.py    FullBatchMLNCELoss
data.py      benchmark loading and split remapping
metrics.py   P-H@K, CP-AUC, CP-AP
train.py     per-seed training loop, multi-seed aggregation, CLI
```
