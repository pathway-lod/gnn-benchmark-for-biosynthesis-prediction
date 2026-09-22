#!/usr/bin/env bash
set -uo pipefail
cd /lustre/BIF/nobackup/delp003/path-graph-ml/plantmetbench/training
export CUDA_VISIBLE_DEVICES=0
COMMON="--gnn-name residual_jumping_sage --num-layers 2 --epochs 650 --early-stop-patience 300 --remove-organism-nodes --no-embeddings --run-name structure_only_baseline"

for seed in 42 0 1; do
  echo "[seed $seed] structure-only baseline (random features, same topology)"
  conda run --no-capture-output -n plantmetbench python train.py $COMMON --seed $seed
done

echo "STRUCTURE_BASELINE_GPU0_DONE"
