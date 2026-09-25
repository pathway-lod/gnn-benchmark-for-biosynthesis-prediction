#!/usr/bin/env bash
set -euo pipefail
cd /lustre/BIF/nobackup/delp003/path-graph-ml/plantmetbench
export CUDA_VISIBLE_DEVICES=0

echo "=== Pathway reconstruction, full taxa-holdout dataset (8,445-protein pool), 5 seeds ==="
/lustre/BIF/nobackup/delp003/miniforge3/bin/conda run --no-capture-output -n plantmetbench \
  python scripts/eval_pathway_reconstruction.py \
  --runs-dir runs/splits_comparison_taxa/residual_jumping_sage_L2_h128_dot_lr0.0001 \
  --seeds 42 0 1 2 3 \
  --tag taxa

echo "FULL_PATHWAY_RECONSTRUCTION_DONE"
