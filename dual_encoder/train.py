"""Train and evaluate the dual-encoder retrieval model on PlantMetBench.

Training is full-batch: every epoch encodes all training reactions and the entire
protein pool, forms the dense reaction x protein distance matrix, and takes a single
optimiser step on the MLNCE objective. The pool therefore acts as the negative set,
so no negative sampling is needed at training time.

The script runs one training job per seed and reports the mean and sample standard
deviation of the test metrics across seeds.

Usage
-----
    python train.py                                  # 5 seeds, default hyperparameters
    python train.py --seeds 42 --epochs 50           # quick single-seed run
    python train.py --data_dir /path/to/plantmetbench
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch

from data import load_plantmet
from losses import FullBatchMLNCELoss
from metrics import encode_all, retrieval_metrics
from model import DualEncoder, cosine_distances

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORTED_METRICS = ("P-H@10", "P-H@50", "cp_auc", "cp_ap")
METRIC_LABELS = {"P-H@10": "P-H@10", "P-H@50": "P-H@50", "cp_auc": "CP-AUC", "cp_ap": "CP-AP"}


def run_seed(args: argparse.Namespace, data, seed: int, device: torch.device) -> dict:
    """Train one model and return its metrics, parameter count and epoch timing."""
    torch.manual_seed(seed)

    model = DualEncoder(
        query_dim=data.reaction_x.shape[1],
        target_dim=data.protein_x.shape[1],
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    loss_fn = FullBatchMLNCELoss(beta=args.beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Restrict the query side to reactions that actually occur in the training split,
    # and remap the pair indices onto that compact table.
    train_pairs = data.splits["train"]
    query_rows, query_local = torch.unique(train_pairs[0], return_inverse=True)
    train_query_x = data.reaction_x[query_rows].to(device)
    pool_x = data.protein_x.to(device)
    query_local = query_local.to(device)
    target_local = train_pairs[1].to(device)

    best_val_score, best_state, epoch_times = -1.0, None, []

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()

        query_emb, target_emb = model(train_query_x, pool_x)
        loss = loss_fn(cosine_distances(query_emb, target_emb), query_local, target_local)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - start)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(model, data, "val", args, device)
            if val[args.model_selection] > best_val_score:
                best_val_score = val[args.model_selection]
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"    epoch {epoch:3d}/{args.epochs}  loss {loss.item():7.4f}  "
                f"val P-H@10 {val['P-H@10']:.4f}  val P-H@50 {val['P-H@50']:.4f}  "
                f"val CP-AUC {val['cp_auc']:.4f}"
            )

    model.load_state_dict(best_state)  # report the best-validation checkpoint
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_dir / f"dual_encoder_seed{seed}.pt")

    return {
        "seed": seed,
        "num_parameters": model.num_parameters,
        "mean_epoch_time_s": statistics.fmean(epoch_times),
        # Peak tensor memory; excludes the CUDA context and allocator fragmentation,
        # so nvidia-smi will report a few hundred MiB more than this.
        "peak_gpu_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        f"best_val_{args.model_selection}": best_val_score,
        "val": evaluate(model, data, "val", args, device),
        "test": evaluate(model, data, "test", args, device),
    }


def evaluate(model: DualEncoder, data, split: str, args: argparse.Namespace, device: torch.device) -> dict:
    """Score one split's positive pairs against the full protein pool."""
    return retrieval_metrics(
        query_emb=encode_all(model.query_encoder, data.reaction_x, device),
        target_emb=encode_all(model.target_encoder, data.protein_x, device),
        pairs=data.splits[split],
        num_negatives=args.num_negatives,
        seed=args.eval_seed,
    )


def summarise(runs: list[dict]) -> dict:
    """Aggregate per-seed test metrics into mean and sample standard deviation."""
    def _stats(values: list[float]) -> dict:
        return {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    return {
        "num_seeds": len(runs),
        "num_parameters": runs[0]["num_parameters"],
        "peak_gpu_mib": runs[0]["peak_gpu_mib"],
        "mean_epoch_time_s": _stats([r["mean_epoch_time_s"] for r in runs]),
        "test": {m: _stats([r["test"][m] for r in runs]) for m in REPORTED_METRICS},
    }


def print_report(summary: dict, runs: list[dict], args: argparse.Namespace) -> None:
    seeds = ", ".join(str(r["seed"]) for r in runs)
    epoch_time = summary["mean_epoch_time_s"]

    print(f"\n{'=' * 64}")
    print(f"  Dual-encoder on PlantMetBench -- test set, {summary['num_seeds']} seeds ({seeds})")
    print(f"{'=' * 64}")
    print(f"  {'Metric':<10}{'Mean':>10}{'Std':>10}")
    print(f"  {'-' * 30}")
    for metric in REPORTED_METRICS:
        stats = summary["test"][metric]
        print(f"  {METRIC_LABELS[metric]:<10}{stats['mean']:>10.4f}{stats['std']:>10.4f}")
    print(f"  {'-' * 30}")
    print(f"  Model parameters   : {summary['num_parameters']:,}")
    print(f"  Time per epoch     : {epoch_time['mean']:.3f} s +/- {epoch_time['std']:.3f} s")
    if summary["peak_gpu_mib"] is not None:
        print(f"  Peak GPU memory    : {summary['peak_gpu_mib']:,.0f} MiB")
    print(f"  Epochs per seed    : {args.epochs}")
    print(f"  Retrieval pool     : {runs[0]['test']['pool_size']:,} proteins")
    print(f"  Test pairs         : {runs[0]['test']['num_pairs']:,}")


_METRIC_ALIASES = {"CP-AUC": "cp_auc", "CP-AP": "cp_ap"}

def main() -> None:
    args = parse_args()
    args.model_selection = _METRIC_ALIASES.get(args.model_selection, args.model_selection)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    print(f"Device: {device}")

    data = load_plantmet(
        args.data_dir, split_file=args.split_file, deduplicate=not args.keep_duplicates,
        species_pool=args.species_pool, protein_embeddings_path=args.protein_embeddings_path,
    )

    runs = []
    for seed in args.seeds:
        print(f"\n>>> seed {seed}")
        runs.append(run_seed(args, data, seed, device))

    summary = summarise(runs)
    print_report(summary, runs, args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"config": vars(args), "runs": runs, "summary": summary}, indent=2))
    print(f"\nFull per-seed results written to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-encoder contrastive retrieval on PlantMetBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=2048, help="Width of each hidden layer.")
    parser.add_argument("--emb_dim", type=int, default=512, help="Shared embedding dimension.")
    parser.add_argument("--num_layers", type=int, default=2, help="Hidden layers per tower.")
    parser.add_argument("--dropout", type=float, default=0.3)
    # Objective and optimisation
    parser.add_argument("--beta", type=float, default=10.0, help="MLNCE inverse temperature.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # Evaluation
    parser.add_argument("--eval_every", type=int, default=5, help="Epochs between validation passes.")
    parser.add_argument("--model_selection", type=str, default="P-H@50",
                        help="Validation metric used to pick the reported checkpoint.")
    parser.add_argument("--num_negatives", type=int, default=50, help="Pool samples per CP metric positive.")
    parser.add_argument("--eval_seed", type=int, default=0,
                        help="Negative-sampling seed; held fixed so every run shares one protocol.")
    # Run control
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split_file", type=str, default="splits_taxa.pt")
    parser.add_argument("--species_pool", action="store_true",
                        help="Restrict the retrieval pool to split_file's ath_protein_idxs "
                             "(only meaningful with --split_file splits_ath_pathway.pt).")
    parser.add_argument("--keep_duplicates", action="store_true",
                        help="Score all 8,445 accessions instead of the 2,232 distinct ESM vectors.")
    parser.add_argument("--protein_embeddings_path", type=str, default=None,
                        help="Path to an alternate protein embeddings .pt file "
                             "({node_id: tensor}, any dim); defaults to "
                             "data_dir/embeddings_protein.pt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Where per-seed checkpoints are written; empty string disables saving.")
    parser.add_argument("--output", type=str, default="results/results.json")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


if __name__ == "__main__":
    main()
