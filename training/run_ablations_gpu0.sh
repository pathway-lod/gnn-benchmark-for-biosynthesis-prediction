#!/usr/bin/env bash
set -uo pipefail
cd /lustre/BIF/nobackup/delp003/path-graph-ml/plantmetbench/training
export CUDA_VISIBLE_DEVICES=0
COMMON="--gnn-name residual_jumping_sage --num-layers 2 --epochs 650 --early-stop-patience 300"

echo "[1/4] organism_ablation: no_organism (5 seeds)"
conda run --no-capture-output -n plantmetbench python train_seeds.py \
  --run-name organism_ablation_none $COMMON --remove-organism-nodes

echo "[2/4] organism_ablation: random_organism / default (5 seeds) -- also = tab:shortcuts Full baseline"
conda run --no-capture-output -n plantmetbench python train_seeds.py \
  --run-name organism_ablation_random $COMMON

echo "[3/4] shortcuts: +Pathway co-membership (seed 42)"
conda run --no-capture-output -n plantmetbench python train.py \
  --run-name shortcuts_keep_pathways $COMMON --seed 42 --keep-pathways

echo "[4/4] shortcuts: +Reverse catalysis edge (seed 42)"
conda run --no-capture-output -n plantmetbench python train.py \
  --run-name shortcuts_keep_catalyzed_by $COMMON --seed 42 --keep-catalyzed-by

echo "GPU0 QUEUE DONE"
