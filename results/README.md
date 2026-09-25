# Results

This directory collects machine-generated result files referenced in the paper appendix.
Each JSON file is produced by a specific script; run the scripts below to regenerate them.
The generated files are not tracked in git (`results/*` is ignored except this README).

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

The configurations are: baseline (all shortcuts removed), with Pathway co-membership,
with the reverse catalysis edge, without the disjoint train ratio, and with EC one-hot
features. `ablations.json` holds the results of each.

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
