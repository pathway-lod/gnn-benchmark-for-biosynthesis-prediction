#!/usr/bin/env bash
set -uo pipefail
cd /lustre/BIF/nobackup/delp003/path-graph-ml/plantmetbench/training
export CUDA_VISIBLE_DEVICES=1
COMMON="--gnn-name residual_jumping_sage --num-layers 2 --epochs 650 --early-stop-patience 300 --remove-organism-nodes"

for seed in 2 3; do
  echo "[seed $seed] shortcuts: +Pathway co-membership, organism removed"
  conda run --no-capture-output -n plantmetbench python train.py \
    --run-name shortcuts_noorg_keep_pathways $COMMON --seed $seed --keep-pathways

  echo "[seed $seed] shortcuts: +Reverse catalysis edge, organism removed"
  conda run --no-capture-output -n plantmetbench python train.py \
    --run-name shortcuts_noorg_keep_catalyzed_by $COMMON --seed $seed --keep-catalyzed-by

  echo "[seed $seed] shortcuts: without disjoint training, organism removed"
  conda run --no-capture-output -n plantmetbench python train.py \
    --run-name shortcuts_noorg_no_disjoint $COMMON --seed $seed --disjoint-train-ratio 0

  echo "[seed $seed] shortcuts: +EC one-hot features, organism removed"
  conda run --no-capture-output -n plantmetbench python train.py \
    --run-name shortcuts_noorg_ec_features $COMMON --seed $seed --ec-features
done

echo "SHORTCUTS_NOORG_GPU1_DONE"
