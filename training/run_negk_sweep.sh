#!/usr/bin/env bash
set -uo pipefail
cd /lustre/BIF/nobackup/delp003/path-graph-ml/plantmetbench/training
export CUDA_VISIBLE_DEVICES=0
COMMON="--gnn-name residual_jumping_sage --num-layers 2 --epochs 650 --early-stop-patience 300 --seed 42 --remove-organism-nodes"

for k in 1 3 5 10 20; do
  echo "[neg_k=$k] starting"
  conda run --no-capture-output -n plantmetbench python train.py \
    --run-name negk_sweep_k${k} $COMMON --neg-k $k --neg-k-cp $k
done

echo "NEG_K SWEEP DONE"
