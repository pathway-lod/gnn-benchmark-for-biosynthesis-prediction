#!/usr/bin/env python3
"""
Stratifies taxa-holdout val/test P-H@K by whether the target reaction
(Conversion node) already has a catalyzing edge in the TRAINING partition
(a different organism's ortholog/isoenzyme for the same curated reaction) or
is genuinely absent from training.

Motivation: Reaction Tanimoto between train and test/val under the taxa
holdout is NOT 0.00 (train-test 0.10, train-val 0.13; see
Appendix~\ref{app:splits}), because a single curated Conversion node can be
catalyzed by orthologous enzymes from several species. This script quantifies
how much easier evaluation is on those "bridging" reactions versus reactions
that are entirely absent from training -- filling the dangling
\ref{app:generalization} reference in sections/appendix.tex.

No retraining, no graph modification: only WHICH positive (protein,
conversion) pairs are scored is changed, exactly like the message-passing
graph and the model see it during normal evaluation. Uses the existing best
checkpoints (HeteroSAGE Res&Jump, splits_comparison_taxa, 5 seeds).

Usage (from repo root):
    python scripts/eval_reaction_overlap.py --seeds 42 0 1 2 3
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "training"))

from build_clean_ctx import load_ctx_for_run
from models import build_model
from metrics import protein_hits_at_k

DATA_DIR = REPO / "data"
RUNS_DIR = REPO / "runs" / "splits_comparison_taxa" / "residual_jumping_sage_L2_h128_dot_lr0.0001"


def positives(data, target_edge):
    lbl = data[target_edge].edge_label
    return data[target_edge].edge_label_index[:, lbl == 1]


def filtered_data(data, target_edge, keep_mask, pos_ei):
    """Clone data with edge_label_index restricted to pos_ei[:, keep_mask]
    (message-passing edge_index_dict is untouched)."""
    d = data.clone()
    kept = pos_ei[:, keep_mask]
    d[target_edge].edge_label_index = kept
    d[target_edge].edge_label = torch.ones(kept.shape[1], dtype=d[target_edge].edge_label.dtype)
    return d


def evaluate(model, ctx, data, k_list):
    return protein_hits_at_k(model.gnn, model.predictor, data, ctx,
                              k_list=k_list, eval_embedded_only=True, any_catalyst_lookup=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 0, 1, 2, 3])
    ap.add_argument("--k-list", nargs="+", type=int, default=[1, 5, 10, 50])
    args = ap.parse_args()

    results = {"val": {"bridging": [], "novel": []}, "test": {"bridging": [], "novel": []}}
    n_bridging = n_novel = None

    for seed in args.seeds:
        ckpt_dir = RUNS_DIR / f"seed_{seed}"
        ckpt_files = sorted(p for p in ckpt_dir.glob("*best*.pt") if "superseded" not in str(p))
        assert ckpt_files, f"no checkpoint in {ckpt_dir}"
        ckpt_path = ckpt_files[-1]
        print(f"\n=== seed {seed}: {ckpt_path.name} ===")
        ckpt = torch.load(ckpt_path, weights_only=False)
        hp = ckpt.get("hparams", {})

        ctx = load_ctx_for_run(hp, DATA_DIR, seed=seed, download=False, print_summary=False)
        target_edge = ctx.target_edge

        # A reaction counts as "in training" if it has a catalyzing edge either
        # in the MP graph (80% of train) or in the disjoint supervision-only
        # labels (the other 20%) -- both represent a training-organism catalyst
        # documented for that Conversion node, not just the MP-graph subset.
        mp_convs = set(ctx.train_data[target_edge].edge_index[1].tolist())
        sup_convs = set(positives(ctx.train_data, target_edge)[1].tolist())
        train_convs = mp_convs | sup_convs

        model = build_model(ctx, gnn_name=hp.get("gnn_name", "sage"),
                             hidden_dim=hp.get("hidden_dim", 128), num_layers=hp.get("num_layers", 2),
                             decoder=hp.get("decoder", "dot"), dropout=hp.get("dropout", 0.3),
                             random_seed=seed, use_norm=hp.get("layer_norm", False))
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        for split_name, data in (("val", ctx.val_data), ("test", ctx.test_data)):
            pos_ei = positives(data, target_edge)
            is_bridging = torch.tensor([c.item() in train_convs for c in pos_ei[1]], dtype=torch.bool)
            if seed == args.seeds[0]:
                n = is_bridging.numel()
                print(f"  {split_name}: {int(is_bridging.sum())}/{n} bridging "
                      f"({int(is_bridging.sum())/n*100:.1f}%), "
                      f"{int((~is_bridging).sum())}/{n} novel")

            bridging_data = filtered_data(data, target_edge, is_bridging, pos_ei)
            novel_data    = filtered_data(data, target_edge, ~is_bridging, pos_ei)

            res_b = evaluate(model, ctx, bridging_data, args.k_list)
            res_n = evaluate(model, ctx, novel_data, args.k_list)
            print(f"    bridging P-H@50={res_b['P-H@50']*100:.2f}%  novel P-H@50={res_n['P-H@50']*100:.2f}%")

            results[split_name]["bridging"].append(res_b)
            results[split_name]["novel"].append(res_n)

    def summarize(entries, metric):
        vals = [r[metric] * 100 for r in entries]
        return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    print(f"\n{'=' * 70}\nSummary across {len(args.seeds)} seeds\n{'=' * 70}")
    for k in args.k_list:
        metric = f"P-H@{k}"
        for split_name in ("val", "test"):
            bm, bs = summarize(results[split_name]["bridging"], metric)
            nm, ns = summarize(results[split_name]["novel"], metric)
            print(f"{split_name:<5} {metric:<8} bridging {bm:6.2f}+/-{bs:4.2f}   "
                  f"novel {nm:6.2f}+/-{ns:4.2f}   gap {bm-nm:+.2f}pp")


if __name__ == "__main__":
    main()
