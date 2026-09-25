#!/usr/bin/env bash
cd /lustre/BIF/nobackup/delp003/path-graph-ml/plantmetbench/training
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n plantmetbench python train_seeds.py \
  --run-name splits_comparison_random --gnn-name residual_jumping_sage --num-layers 2 \
  --remove-organism-nodes --split-type random --epochs 650 --early-stop-patience 300
