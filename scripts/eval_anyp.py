#!/usr/bin/env python3
"""Evaluate anyP-H@K on a saved best checkpoint.

anyP-H@K counts a reaction as a hit if ANY known catalyst of that reaction
(across train, val, and test splits) appears in the top-K ranked proteins.
This removes false penalties for correctly ranking isoenzymes or orthologues.

Usage (from repo root):
    python scripts/eval_anyp.py --run-name sage_1layer/seed_42
    python scripts/eval_anyp.py --run-name sage_1layer/seed_42 --split val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from build_clean_ctx import load_ctx_for_run
from metrics import protein_hits_at_k
from models import build_model
from utils import set_seed

RUNS_DIR = Path(__file__).parent.parent / "runs"


def get_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True,
                    help="Run directory under runs/ (e.g. sage_1layer/seed_42)")
    ap.add_argument("--split", choices=["val", "test", "both"], default="both")
    ap.add_argument("--data-dir", default=str(Path(__file__).parent.parent / "data"))
    ap.add_argument("--k-list", default="1,5,10,50",
                    help="Comma-separated K values (default: 1,5,10,50)")
    ap.add_argument("--num-layers", type=int, default=None,
                    help="Override num_layers (inferred from checkpoint hparams if omitted)")
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--remove-organism-nodes", action="store_true", default=False)
    return ap.parse_args()


def main():
    args = get_args()
    k_list = tuple(int(k) for k in args.k_list.split(","))

    run_dir = RUNS_DIR / args.run_name
    ckpts   = sorted(run_dir.glob("*_best.pt"))
    if not ckpts:
        print(f"ERROR: no *_best.pt checkpoint in {run_dir}")
        sys.exit(1)

    ckpt_data = torch.load(ckpts[0], weights_only=False)
    hp = ckpt_data.get("hparams", {})
    num_layers = args.num_layers or hp.get("num_layers", 2)
    hidden_dim = args.hidden_dim  or hp.get("hidden_dim", 128)
    seed       = hp.get("seed", 42)
    remove_org = args.remove_organism_nodes or hp.get("remove_organism_nodes", False)

    set_seed(seed)
    print(f"Run      : {args.run_name}")
    print(f"Layers   : {num_layers}  hidden={hidden_dim}  seed={seed}")
    print(f"Organisms: {'removed' if remove_org else 'included'}")
    print()

    ctx = load_ctx_for_run(hp, args.data_dir, remove_organism_nodes=remove_org, download=False)

    model = build_model(ctx, gnn_name=hp.get("gnn_name", "sage"),
                        hidden_dim=hidden_dim, num_layers=num_layers,
                        decoder=hp.get("decoder", "dot"), dropout=hp.get("dropout", 0.3),
                        random_seed=seed, use_norm=hp.get("layer_norm", False))
    model.load_state_dict(ckpt_data["model_state"])
    model.eval()

    splits = []
    if args.split in ("val", "both"):
        splits.append(("val", ctx.val_data))
    if args.split in ("test", "both"):
        splits.append(("test", ctx.test_data))

    print(f"{'Split':<6}  {'Metric':<14}  {'Standard P-H@K':>16}  {'anyP-H@K':>12}")
    print("─" * 56)
    for split_name, data in splits:
        standard = protein_hits_at_k(
            model.gnn, model.predictor, data, ctx,
            k_list=k_list, eval_embedded_only=True, any_catalyst_lookup=None,
        )
        anyp = protein_hits_at_k(
            model.gnn, model.predictor, data, ctx,
            k_list=k_list, eval_embedded_only=True,
            any_catalyst_lookup=ctx.all_catalyst_lookup,
        )
        for k in k_list:
            std_v  = standard.get(f"P-H@{k}", float("nan"))
            any_v  = anyp.get(f"anyP-H@{k}", float("nan"))
            print(f"{split_name:<6}  P-H@{k:<10}  {std_v*100:>15.2f}%  {any_v*100:>11.2f}%")
        print()

    # Also print how many reactions have >1 known catalyst
    n_multi = sum(1 for s in ctx.all_catalyst_lookup.values() if len(s) > 1)
    n_total = len(ctx.all_catalyst_lookup)
    print(f"Catalyst lookup: {n_total} reactions total, "
          f"{n_multi} ({n_multi/n_total*100:.1f}%) have >1 known catalyst")


if __name__ == "__main__":
    main()
