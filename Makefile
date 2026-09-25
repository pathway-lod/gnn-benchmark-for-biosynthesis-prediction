# Entry points for the experiments in the paper. Every target is a thin wrapper
# around a documented command, so `make -n <target>` shows exactly what it runs.
#
#   make help
#
# Variables:  PYTHON=path/to/python   SEEDS=42,0,1,2,3   GNN=sage   LAYERS=2   RUN=baseline

PYTHON ?= python
SEEDS  ?= 42,0,1,2,3
GNN    ?= sage
LAYERS ?= 2
RUN    ?= baseline

.PHONY: help data baseline seeds model ablations ec-distribution train-frac \
        ath-splits ath dual-encoder dual-encoder-fullpool dual-encoder-dedup blast blast-dedup tables

help:
	@echo "Setup"
	@echo "  data                download the dataset from Zenodo into data/"
	@echo "GNN experiments"
	@echo "  baseline            one training run, seed 42 (train.py defaults)"
	@echo "  seeds               5-seed run of the baseline (train_seeds.py)"
	@echo "  model               5-seed run of one model: make model GNN=residual_jumping_sage LAYERS=2 RUN=resjump"
	@echo "  ablations           shortcut ablations -> results/ablations.json"
	@echo "  ec-distribution     EC class distribution -> results/ec_distribution.json"
	@echo "  train-frac          data-scaling ablation (train_frac_ablation.py)"
	@echo "  ath-splits          build the A. thaliana reaction-holdout split"
	@echo "  ath                 run all models on the A. thaliana reaction holdout"
	@echo "Baselines"
	@echo "  dual-encoder        dual encoder, default 2,232-protein pool (5 seeds)"
	@echo "  dual-encoder-fullpool  dual encoder, 8,445-protein pool (--keep_duplicates), L2"
	@echo "  dual-encoder-dedup  dual encoder L1-L3 on the 2,232-protein pool"
	@echo "  blast               BLASTp baseline, test split, 8,445-protein pool"
	@echo "  blast-dedup         BLASTp baseline on the 2,232-protein pool"
	@echo "Reporting"
	@echo "  tables              print the appendix tables from results/"

data:
	$(PYTHON) training/dataset.py

baseline:
	$(PYTHON) training/train.py

seeds:
	$(PYTHON) training/train_seeds.py --run-name $(RUN) --seeds $(SEEDS)

model:
	$(PYTHON) training/train_seeds.py --run-name $(RUN) --seeds $(SEEDS) --gnn-name $(GNN) --num-layers $(LAYERS)

ablations:
	cd training && $(PYTHON) ../scripts/run_ablations.py

ec-distribution:
	cd training && $(PYTHON) ../scripts/extract_ec_distribution.py

train-frac:
	$(PYTHON) training/train_frac_ablation.py

ath-splits:
	$(PYTHON) training/build_splits_ath.py

ath:
	bash scripts/run_ath_baseline.sh

dual-encoder:
	cd baselines/dual_encoder && $(PYTHON) train.py

dual-encoder-fullpool:
	cd baselines/dual_encoder && $(PYTHON) train.py --seeds 42 0 1 2 3 --model_selection P-H@50 --num_layers 2 \
	  --keep_duplicates --checkpoint_dir checkpoints_fullpool --output results/taxa_L2_ph50_fullpool.json

dual-encoder-dedup:
	bash baselines/dual_encoder/run_dedup.sh

blast:
	$(PYTHON) baselines/blast/blast_evaluate.py

blast-dedup:
	$(PYTHON) baselines/blast/blast_evaluate.py --dedup-pool

tables:
	$(PYTHON) scripts/render_results_tables.py
