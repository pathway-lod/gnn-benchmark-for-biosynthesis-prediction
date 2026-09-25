#!/usr/bin/env bash
# Dual-encoder on the A. thaliana within-species split, restricted to the
# 5,289-protein ESM-C pool (--species_pool), matching the GNN evaluation.
set -euo pipefail
cd /lustre/BIF/nobackup/delp003/path-graph-ml
export CUDA_VISIBLE_DEVICES=1
RESULTS=plantmetbench/results/dual_encoder
mkdir -p "$RESULTS"

/lustre/BIF/nobackup/delp003/miniforge3/bin/conda run --no-capture-output -n plantmetbench \
  python plantmetbench/dual_encoder/train.py \
  --seeds 42 0 1 2 3 --model_selection P-H@50 --num_layers 2 \
  --split_file splits_ath_pathway.pt --species_pool \
  --output "$RESULTS/ath_L2_ph50.json"

echo "DUAL_ENCODER_ATH DONE"
