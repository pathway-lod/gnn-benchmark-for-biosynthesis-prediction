#!/usr/bin/env bash
set -uo pipefail
cd /lustre/BIF/nobackup/delp003/path-graph-ml/plantmetbench
export CUDA_VISIBLE_DEVICES=1
CFG=runs/ath_resjump/residual_jumping_sage_L2_h128_dot_lr0.0001
OUT=/tmp/claude-17498538/-lustre-BIF-nobackup-delp003-path-graph-ml/121118d3-4feb-4781-a51c-247d5cd36f22/scratchpad/ath_eval

mkdir -p "$OUT"

echo "=== [1/3] anyP-H@K per seed ==="
for seed in 42 0 1 2 3; do
  echo "-- seed $seed --"
  conda run --no-capture-output -n plantmetbench python scripts/eval_anyp.py \
    --run-name ath_resjump/residual_jumping_sage_L2_h128_dot_lr0.0001/seed_$seed \
    --split both > "$OUT/anyp_seed${seed}.log" 2>&1
done
echo "anyP done"

echo "=== [2/3] pathway reconstruction (seeds 42 0 1) ==="
conda run --no-capture-output -n plantmetbench python scripts/eval_pathway_reconstruction.py \
  --runs-dir "$CFG" --seeds 42 0 1 --no-plot > "$OUT/pathway_recon.log" 2>&1
cp results/pathway_reconstruction.json "$OUT/pathway_reconstruction.json"
echo "pathway reconstruction done"

echo "=== [3/3] Rhea validation (seed 42) ==="
conda run --no-capture-output -n plantmetbench python scripts/eval_rhea_validation.py \
  --run-dir "$CFG/seed_42" --rhea-dir data/rhea_cache --split both > "$OUT/rhea.log" 2>&1
echo "Rhea done"

echo "ATH_EVAL ALL DONE"
