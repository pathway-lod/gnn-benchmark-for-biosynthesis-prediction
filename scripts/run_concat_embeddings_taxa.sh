#!/usr/bin/env bash
set -euo pipefail
cd /lustre/BIF/nobackup/delp003/path-graph-ml/plantmetbench/training
export CUDA_VISIBLE_DEVICES=0
CONDA=/lustre/BIF/nobackup/delp003/miniforge3/bin/conda

for seed in 42 0 1 2 3; do
  echo "=== seed $seed ==="
  $CONDA run --no-capture-output -n plantmetbench python train.py \
    --gnn-name residual_jumping_sage --num-layers 2 \
    --remove-organism-nodes \
    --protein-embeddings ../data/protein_embeddings_alt/embeddings_protein_esm2mean_esmcgraphec_pagerank_concat.pt \
    --epochs 650 --early-stop-patience 300 --seed $seed \
    --run-name protein_emb_concat_taxa
done

echo "PROTEIN_EMB_CONCAT_TAXA_DONE"
