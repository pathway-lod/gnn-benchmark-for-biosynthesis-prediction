# Results

This directory collects machine-generated result files referenced in the paper appendix.
Each JSON file is produced by a specific script; run the scripts below to regenerate them.

---

## Files

| File | Appendix table | How to generate |
|---|---|---|
| `ec_distribution.json` | Table B.4 | `python scripts/extract_ec_distribution.py` |
| `ablations.json` | Table E.1 | `python scripts/run_ablations.py` |
| `baseline/seeds_summary.json` | Table G.4 | `python training/train_seeds.py` |

---

## Step-by-step instructions

All commands are run from the **repo root** (`plantmetbench/`).

### 0. Setup

```bash
conda env create -f environment.yml
conda activate plantmetbench
```

Data is downloaded automatically on first run. You can also pre-download:

```bash
python training/dataset.py   # downloads heterodata.pt + embeddings from Zenodo
```

---

### Table B.4 — EC Class Distribution

```bash
python scripts/extract_ec_distribution.py
```

Reads `data/embeddings_ec.pt` and writes `results/ec_distribution.json`.

---

### Table E.1 — Shortcut Ablations

Run all 5 ablation experiments (baseline + 4 shortcuts re-enabled one at a time):

```bash
cd training/
python ../scripts/run_ablations.py
```

Each experiment is ~200 epochs and runs sequentially. Writes `results/ablations.json`.

To re-run only one ablation (e.g., the EC features shortcut):

```bash
python ../scripts/run_ablations.py --only ec_features
```

To skip experiments already completed:

```bash
python ../scripts/run_ablations.py --skip-existing
```

Expected results (baseline, seed=42):

| Configuration | Val P-H@50 | Test P-H@50 |
|---|---|---|
| Baseline (all shortcuts removed) | ~11.5% | ~6.0% |
| +Pathway co-membership | TBD | TBD |
| +Reverse catalysis edge | TBD | TBD |
| No disjoint train ratio | TBD | TBD |
| +EC one-hot features | TBD | TBD |

---

### Table G.4 — 5-Seed Mean ± Std

```bash
cd training/
python train_seeds.py --run-name baseline
```

Runs seeds `42, 0, 1, 2, 3` sequentially. Results go to `runs/baseline/` and the
summary is written to `results/baseline/seeds_summary.json` as well as
`runs/baseline/seeds_summary.json`.

Estimated wall time: ~5 × training time per run.

---

### GPU Timing

Training time and peak GPU memory are automatically logged to `report.json` in each
run directory under `runs/`. After any training run:

```bash
cat runs/baseline/seed_42/report.json | python -m json.tool | grep -A5 compute_stats
```

---

## JSON schemas

### `ec_distribution.json`

```json
{
  "n_conversion_nodes": 19927,
  "n_annotated": 11234,
  "n_unannotated": 8693,
  "annotation_coverage_pct": 56.4,
  "counts_by_ec_class": {
    "Oxidoreductases": ...,
    "Transferases": ...,
    ...
    "Unknown / unannotated": ...
  }
}
```

### `ablations.json`

```json
{
  "ablations": [
    {
      "name": "baseline",
      "description": "All shortcuts removed (default baseline)",
      "flags": [],
      "results": { "val_ph50": ..., "test_ph50": ..., ... }
    },
    ...
  ]
}
```

### `baseline/seeds_summary.json`

```json
{
  "seeds": [42, 0, 1, 2, 3],
  "metrics": {
    "val_ph50":  { "mean": ..., "std": ... },
    "test_ph50": { "mean": ..., "std": ... },
    ...
  }
}
```
