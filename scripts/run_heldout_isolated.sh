#!/usr/bin/env bash
set -euo pipefail
cd /lustre/BIF/nobackup/delp003/path-graph-ml/plantmetbench
export CUDA_VISIBLE_DEVICES=0
/lustre/BIF/nobackup/delp003/miniforge3/bin/conda run --no-capture-output -n plantmetbench \
  python scripts/eval_heldout_isolated.py --seeds 42 0 1 2 3
echo "HELDOUT_ISOLATED_DONE"
