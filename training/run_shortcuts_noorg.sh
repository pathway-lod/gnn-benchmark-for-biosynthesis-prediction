#!/usr/bin/env bash
set -uo pipefail
cd /lustre/BIF/nobackup/delp003/path-graph-ml/plantmetbench/training
export CUDA_VISIBLE_DEVICES=0
COMMON="--gnn-name residual_jumping_sage --num-layers 2 --epochs 650 --early-stop-patience 300 --seed 42 --remove-organism-nodes"

echo "[1/4] shortcuts: +Pathway co-membership, organism removed (seed 42)"
conda run --no-capture-output -n plantmetbench python train.py \
  --run-name shortcuts_noorg_keep_pathways $COMMON --keep-pathways

echo "[2/4] shortcuts: +Reverse catalysis edge, organism removed (seed 42)"
conda run --no-capture-output -n plantmetbench python train.py \
  --run-name shortcuts_noorg_keep_catalyzed_by $COMMON --keep-catalyzed-by

echo "[3/4] shortcuts: without disjoint training, organism removed (seed 42)"
conda run --no-capture-output -n plantmetbench python train.py \
  --run-name shortcuts_noorg_no_disjoint $COMMON --disjoint-train-ratio 0

echo "[4/4] shortcuts: +EC one-hot features, organism removed (seed 42)"
conda run --no-capture-output -n plantmetbench python train.py \
  --run-name shortcuts_noorg_ec_features $COMMON --ec-features

echo "SHORTCUTS_NOORG DONE"
