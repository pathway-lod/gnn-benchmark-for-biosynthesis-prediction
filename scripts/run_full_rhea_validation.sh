#!/usr/bin/env bash
set -uo pipefail
cd /lustre/BIF/nobackup/delp003/path-graph-ml/plantmetbench
export CUDA_VISIBLE_DEVICES=1
CFG=runs/splits_comparison_taxa/residual_jumping_sage_L2_h128_dot_lr0.0001
OUT=results/rhea_taxa
mkdir -p "$OUT"

echo "=== Rhea validation, full taxa-holdout dataset (8,445-protein pool), 5 seeds ==="
for seed in 42 0 1 2 3; do
  echo "-- seed $seed --"
  /lustre/BIF/nobackup/delp003/miniforge3/bin/conda run --no-capture-output -n plantmetbench \
    python scripts/eval_rhea_validation.py \
    --run-dir "$CFG/seed_$seed" \
    --rhea-dir data/rhea_cache \
    --split both \
    2>&1 | tee "$OUT/rhea_seed${seed}.log"
done

echo "FULL_RHEA_VALIDATION_DONE"
