#!/usr/bin/env bash
set -euo pipefail
cd /lustre/BIF/nobackup/delp003/path-graph-ml/plantmetbench
export CUDA_VISIBLE_DEVICES=0
RESULTS=results/dual_encoder
mkdir -p "$RESULTS"

/lustre/BIF/nobackup/delp003/miniforge3/bin/conda run --no-capture-output -n plantmetbench \
  python dual_encoder/train.py \
  --seeds 42 0 1 2 3 --model_selection P-H@50 --num_layers 2 \
  --keep_duplicates \
  --protein_embeddings_path data/protein_embeddings_alt/embeddings_protein_esm2mean_esmcgraphec_pagerank_concat.pt \
  --checkpoint_dir checkpoints_concat_fullpool \
  --output "$RESULTS/taxa_concat_fullpool.json"

echo "DUAL_ENCODER_CONCAT_FULLPOOL_DONE"
