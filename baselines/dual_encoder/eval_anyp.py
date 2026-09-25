"""Evaluate anyP-H@K for saved dual-encoder checkpoints on PlantMetBench.

anyP-H@K counts a query reaction as a hit if ANY known catalyst of that
reaction -- across train, val and test of the loaded split -- appears in the
top-K ranked pool candidates, mirroring
scripts/eval_anyp.py / training/dataset.py's all_catalyst_lookup for the GNN.

Usage (from repo root):
    python baselines/dual_encoder/eval_anyp.py \
        --checkpoint_dir checkpoints_fullpool --split_file splits_ath_pathway.pt \
        --species_pool --keep_duplicates
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from data import load_plantmet
from metrics import encode_all, retrieval_metrics
from model import DualEncoder

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_KS = (1, 5, 10, 50)


def build_all_catalyst_lookup(data) -> dict[int, set[int]]:
    lookup: dict[int, set[int]] = defaultdict(set)
    for split in ("train", "val", "test"):
        pairs = data.splits[split]
        for q, t in zip(pairs[0].tolist(), pairs[1].tolist()):
            lookup[q].add(t)
    return lookup


@torch.no_grad()
def anyp_hits_at_k(query_emb, target_emb, split_pairs, all_catalysts, ks=DEFAULT_KS):
    unique_rxns = sorted(set(split_pairs[0].tolist()))
    scores_all = query_emb[unique_rxns] @ target_emb.t()  # [R, P]
    hits = {k: 0 for k in ks}
    for row, rxn in enumerate(unique_rxns):
        known = all_catalysts.get(rxn, set())
        if not known:
            continue
        order = torch.argsort(scores_all[row], descending=True)
        for k in ks:
            top_k = set(order[:k].tolist())
            if top_k & known:
                hits[k] += 1
    n = sum(1 for r in unique_rxns if all_catalysts.get(r))
    return {f"anyP-H@{k}": hits[k] / n for k in ks}, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    ap.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--split_file", type=str, default="splits_taxa.pt")
    ap.add_argument("--species_pool", action="store_true")
    ap.add_argument("--keep_duplicates", action="store_true")
    ap.add_argument("--emb_dim", type=int, default=512)
    ap.add_argument("--hidden_dim", type=int, default=2048)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_negatives", type=int, default=50)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--split", choices=["val", "test", "both"], default="both")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    data = load_plantmet(
        args.data_dir, split_file=args.split_file,
        deduplicate=not args.keep_duplicates, species_pool=args.species_pool,
    )
    all_catalysts = build_all_catalyst_lookup(data)

    splits_to_eval = ["val", "test"] if args.split == "both" else [args.split]

    per_seed: dict[str, list[dict]] = {s: [] for s in splits_to_eval}
    for seed in args.seeds:
        ckpt_path = Path(args.checkpoint_dir) / f"dual_encoder_seed{seed}.pt"
        model = DualEncoder(
            query_dim=data.reaction_x.shape[1], target_dim=data.protein_x.shape[1],
            emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        query_emb = encode_all(model.query_encoder, data.reaction_x, device)
        target_emb = encode_all(model.target_encoder, data.protein_x, device)

        print(f"\n>>> seed {seed}  ({ckpt_path})")
        for split in splits_to_eval:
            std = retrieval_metrics(query_emb, target_emb, data.splits[split],
                                     num_negatives=args.num_negatives, seed=args.eval_seed)
            anyp, n_any = anyp_hits_at_k(query_emb, target_emb, data.splits[split], all_catalysts)
            merged = {**std, **anyp, "n_any_reactions": n_any}
            per_seed[split].append(merged)
            print(f"  {split:<5} P-H@50 {std['P-H@50']:.4f}  anyP-H@50 {anyp['anyP-H@50']:.4f}"
                  f"  CP-AUC {std['cp_auc']:.4f}  (n_any={n_any})")

    print(f"\n{'=' * 64}\n  Summary across {len(args.seeds)} seeds ({args.seeds})\n{'=' * 64}")
    for split in splits_to_eval:
        runs = per_seed[split]
        print(f"\n-- {split} --")
        for metric in ("P-H@1", "P-H@5", "P-H@10", "P-H@50",
                        "anyP-H@1", "anyP-H@5", "anyP-H@10", "anyP-H@50",
                        "cp_auc", "cp_ap"):
            vals = [r[metric] for r in runs]
            mean = statistics.fmean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {metric:<10} {mean*100:6.2f} +/- {std*100:5.2f} %")


if __name__ == "__main__":
    main()
