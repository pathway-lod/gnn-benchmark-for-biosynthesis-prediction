#!/usr/bin/env bash
# Run the models on the A. thaliana reaction holdout (split type ath_pathway):
# reactions are held out within A. thaliana, stratified by EC class, and proteins
# are ranked over the A. thaliana protein pool only.
#
# Prerequisites (from the repo root):
#   python training/dataset.py               # download the data
#   python training/build_splits_ath.py      # writes data/splits_ath_pathway.pt
#
# Run from the repo root:
#   bash scripts/run_ath_baseline.sh
#
# Override the python interpreter:
#   PYTHON=/path/to/env/bin/python bash scripts/run_ath_baseline.sh
#
# GNN runs land in runs/ath_baseline/ (train_seeds.py also writes the seed summary
# to results/ath_baseline/); the dual encoder writes to results/ath_baseline/.

set -euo pipefail

PYTHON=${PYTHON:-python}
SEEDS="42,0,1,2,3"
COMMON="--split-type ath_pathway --species-pool"
RESULTS=results/ath_baseline

mkdir -p "$RESULTS"

for GNN in sage gat hgt residual_jumping_sage; do
  echo ""
  echo "--- $GNN ---"
  $PYTHON training/train_seeds.py --run-name ath_baseline --seeds "$SEEDS" \
    --gnn-name "$GNN" $COMMON
done

echo ""
echo "--- Dual encoder (L1) ---"
OUT="$RESULTS/dual_encoder_L1_ath.json"
if [[ -f "$OUT" ]]; then
  echo "  already exists, skipping."
else
  $PYTHON baselines/dual_encoder/train.py \
    --seeds 42 0 1 2 3 \
    --num_layers 1 \
    --split_file splits_ath_pathway.pt \
    --species_pool \
    --output "$OUT"
fi
