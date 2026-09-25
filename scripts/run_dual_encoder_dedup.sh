#!/usr/bin/env bash
# Run dual-encoder experiments on the 2,232 deduplicated protein pool
# (no --keep_duplicates) for L1, L2, L3, model selected by P-H@50.
#
# These fill the "2,232 pool" column in Appendix Table (tab:dedup_results).
# Runs the dual encoder with L1, L2 and L3 on the 2,232-protein deduplicated pool.
#
# Run from repo root:
#   bash plantmetbench/scripts/run_dual_encoder_dedup.sh
#
# Results land in plantmetbench/results/dual_encoder/:
#   L1_dedup_ph50.json  L2_dedup_ph50.json  L3_dedup_ph50.json

set -euo pipefail

PYTHON=${PYTHON:-python}   # override: PYTHON=/path/to/env/bin/python bash run_dual_encoder_dedup.sh
TRAIN=plantmetbench/dual_encoder/train.py
RESULTS=plantmetbench/results/dual_encoder
COMMON="--seeds 42 0 1 2 3 --model_selection P-H@50"

mkdir -p "$RESULTS"

echo "=============================="
echo " Dual encoder — 2,232 pool"
echo "=============================="

echo ""
echo "--- L1 (1 hidden layer) ---"
if [[ -f "$RESULTS/L1_dedup_ph50.json" ]]; then
  echo "  already exists, skipping."
else
  $PYTHON $TRAIN $COMMON --num_layers 1 \
    --output "$RESULTS/L1_dedup_ph50.json"
fi

echo ""
echo "--- L2 (2 hidden layers) ---"
if [[ -f "$RESULTS/L2_dedup_ph50.json" ]]; then
  echo "  already exists, skipping."
else
  $PYTHON $TRAIN $COMMON --num_layers 2 \
    --output "$RESULTS/L2_dedup_ph50.json"
fi

echo ""
echo "--- L3 (3 hidden layers) ---"
if [[ -f "$RESULTS/L3_dedup_ph50.json" ]]; then
  echo "  already exists, skipping."
else
  $PYTHON $TRAIN $COMMON --num_layers 3 \
    --output "$RESULTS/L3_dedup_ph50.json"
fi

echo ""
echo "=============================="
echo " BLASTp — 2,232 pool"
echo "=============================="
echo ""
echo "Run from repo root:"
echo "  python scripts/blast_evaluate.py --dedup-pool"
echo "(BLASTp results are deterministic; output is printed to stdout.)"
echo ""
echo "All done."
