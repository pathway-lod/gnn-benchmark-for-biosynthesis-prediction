#!/usr/bin/env bash
set -uo pipefail
cd /lustre/BIF/nobackup/delp003/path-graph-ml/plantmetbench/training
export CUDA_VISIBLE_DEVICES=0

echo "[1/1] ath_pathway: Res&Jump L2, organism removed, species pool (5 seeds)"
conda run --no-capture-output -n plantmetbench python train_seeds.py \
  --run-name ath_resjump --gnn-name residual_jumping_sage --num-layers 2 \
  --remove-organism-nodes --split-type ath_pathway --species-pool \
  --epochs 650 --early-stop-patience 300

echo "ATH_PATHWAY DONE"
