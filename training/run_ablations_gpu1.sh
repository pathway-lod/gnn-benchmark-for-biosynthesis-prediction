#!/usr/bin/env bash
set -uo pipefail
cd /lustre/BIF/nobackup/delp003/path-graph-ml/plantmetbench/training
export CUDA_VISIBLE_DEVICES=1
COMMON="--gnn-name residual_jumping_sage --num-layers 2 --epochs 650 --early-stop-patience 300"

echo "[1/3] organism_ablation: taxonomy_mds (5 seeds)"
conda run --no-capture-output -n plantmetbench python train_seeds.py \
  --run-name organism_ablation_mds $COMMON \
  --organism-embeddings /lustre/BIF/nobackup/delp003/path-graph-ml/plantmetbench/data/embeddings_organism.pt \
  --organism-embedding-type mds

echo "[2/3] shortcuts: without disjoint training (seed 42)"
conda run --no-capture-output -n plantmetbench python train.py \
  --run-name shortcuts_no_disjoint $COMMON --seed 42 --disjoint-train-ratio 0

echo "[3/3] shortcuts: +EC one-hot features (seed 42)"
conda run --no-capture-output -n plantmetbench python train.py \
  --run-name shortcuts_ec_features $COMMON --seed 42 --ec-features

echo "GPU1 QUEUE DONE"
