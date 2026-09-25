#!/usr/bin/env python3
"""Train a heterogeneous GNN for enzyme–reaction link prediction on PlantMetBench.

Task: predict which protein (enzyme) catalyses a given biochemical reaction.
Split: taxa holdout — val/test organisms are held out from training.
Graph: PlantMetWiki knowledge graph, 424 plant species.

Baseline (all defaults):
  HeteroSAGE · 2 layers · 128-dim · per-layer LayerNorm · dot decoder · 200 epochs · seed 42
  Taxa split · cleaned graph (alias Metabolite/GeneProduct and blank-subtype Interaction nodes dropped)
  remove Pathway edges · remove (Metabolite, organism, Organism) edges
  disjoint_train_ratio=0.2 · no catalyzed_by
  MAP4 conversion fingerprints (3072-dim) · ESM-C protein embeddings (960-dim)
  Pre-merge configuration: --data-type raw --no-layer-norm --no-remove-metabolite-organism-edges

Metrics:
  P-H@50   PRIMARY   fraction of reactions where true catalyst is in top-50 of 8,445
  CP-AUC   SECONDARY AUC for true (Protein, Reaction) vs random-protein pair
  CP-AP    SECONDARY average precision for the same task
  Random P-H@50 ≈ 50/8445 = 0.59%

Quick start:
  python train.py                                   # baseline
  python train.py --hidden-dim 256 --num-layers 3   # larger model
  python train.py --no-embeddings                   # structural baseline (no ESM/MAP4)
  python train_seeds.py                             # 5-seed run → seeds_summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from build_clean_ctx import load_ctx_for_run
from metrics import evaluate_cp_auc, evaluate_random_neg_auc, protein_hits_at_k
from models import build_model
from report import TeeLogger, collect_dataset_stats, collect_model_stats, save_report
from utils import collect_all_pairs, set_seed, sample_training_negatives

RUNS_DIR = Path(__file__).parent.parent / "runs"


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    ap.add_argument("--data-dir", default=str(Path(__file__).parent.parent / "data"))
    ap.add_argument("--no-download", dest="download", action="store_false", default=True)

    # Model architecture
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--decoder", choices=["dot", "mlp"], default="dot")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-embeddings", dest="use_embeddings", action="store_false", default=True,
                    help="Replace pre-computed embeddings with random features (structural baseline)")

    # Graph ablations (shortcut control)
    ap.add_argument("--disjoint-train-ratio", type=float, default=0.2,
                    help="Fraction of train positives withheld from MP graph (default 0.2). "
                         "Set to 0 to disable (re-introduces 1-hop shortcut).")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="Fraction of taxa-split training positives to keep (default 1.0). "
                         "Val/test are untouched. Use <1.0 to build a data-scaling learning "
                         "curve (see train_frac_ablation.py).")
    ap.add_argument("--keep-pathways", dest="keep_pathways", action="store_true", default=False,
                    help="Keep is_part_of Pathway edges in the MP graph (re-introduces co-membership shortcut).")
    ap.add_argument("--keep-catalyzed-by", dest="keep_catalyzed_by", action="store_true", default=False,
                    help="Keep (Interaction, catalyzed_by, Protein) reverse edges (re-introduces 2-hop shortcut).")
    ap.add_argument("--ec-features", dest="ec_features", action="store_true", default=False,
                    help="Append 237-dim EC one-hot to Interaction nodes. "
                         "WARNING: causes ~80%% val→test gap — ablation only.")
    ap.add_argument("--split-type", dest="split_type",
                    choices=["taxa", "pathway", "random", "ath_pathway"], default="taxa",
                    help="Holdout strategy: 'taxa' (species holdout, default), "
                         "'pathway' (pathway holdout), 'random' (random edge split), "
                         "or 'ath_pathway' (A. thaliana within-species pathway holdout).")
    ap.add_argument("--species-pool", dest="species_pool",
                    action="store_true", default=False,
                    help="Restrict ranking pool to species proteins (only with ath_pathway).")
    ap.add_argument("--remove-gene-organism-edges", dest="remove_gene_organism_edges",
                    action="store_true", default=False)
    ap.add_argument("--remove-organism-nodes", dest="remove_organism_nodes",
                    action="store_true", default=False,
                    help="Remove all Organism nodes and edges from the MP graph "
                         "(pure-biochemistry baseline, no species topology).")
    ap.add_argument("--remove-currency-metabolites", dest="remove_currency_metabolites",
                    action="store_true", default=False)
    ap.add_argument("--remove-all-metabolites", dest="remove_all_metabolites",
                    action="store_true", default=False)
    ap.add_argument("--organism-embeddings", dest="organism_embeddings_path",
                    default=None, metavar="PATH",
                    help="Path to embeddings_organism.pt for taxonomy-aware Organism features")
    ap.add_argument("--organism-embedding-type", dest="organism_embedding_type",
                    choices=["mds", "multihot"], default="mds")
    ap.add_argument("--protein-embeddings", dest="protein_embeddings_path",
                    default=None, metavar="PATH",
                    help="Path to an alternate protein embeddings .pt file "
                         "({node_id: tensor}, any dim); defaults to "
                         "data_dir/embeddings_protein.pt")

    # Training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=0.5,
                    help="Max gradient norm (clip_grad_norm_); 0 disables clipping (default 1.0)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg-k", type=int, default=5,
                    help="(Protein, Conversion_random) negatives per positive (default 5)")
    ap.add_argument("--neg-k-cp", type=int, default=5,
                    help="(Protein_random, Conversion) negatives per positive (default 5). "
                         "Required for healthy CP-AUC — trains the protein-ranking direction.")
    ap.add_argument("--early-stop-patience", type=int, default=300,
                    help="Stop if val P-H@50 has not improved in this many epochs (0=disable)")
    ap.add_argument("--lr-scheduler", choices=["none", "cosine", "plateau", "step"], default="none",
                    help="LR schedule: 'cosine' (anneal to 0 over --epochs), "
                         "'plateau' (halve lr when val_loss stalls for --lr-patience epochs), "
                         "'step' (multiply by --lr-gamma every --lr-step-size epochs), "
                         "or 'none' (default, constant lr)")
    ap.add_argument("--lr-step-size", type=int, default=50,
                    help="Epochs between decays for --lr-scheduler step")
    ap.add_argument("--lr-gamma", type=float, default=0.5,
                    help="Decay factor for --lr-scheduler step/plateau")
    ap.add_argument("--lr-patience", type=int, default=30,
                    help="Epochs without val_loss improvement before --lr-scheduler plateau decays")
    ap.add_argument("--neg-pool", choices=["all", "catalyzed"], default="all",
                    help="Pool of Conversion reactions to draw (Protein, Conversion_random) training "
                         "negatives from. 'all' (default): the full ~19,927-reaction Conversion pool, "
                         "~77%% of which have no curated catalyst at all — sampling one of those as a "
                         "negative is a closed-world assumption that may just be a missing label, not "
                         "a true negative. 'catalyzed': restrict to the ~4,572 reactions that do have a "
                         "known catalyst, removing that ambiguity (evaluation is unaffected either way, "
                         "since it only ever queries catalyzed reactions).")

    # Logging
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--run-name", type=str, default="baseline",
                    help="Name for this run (used in runs/<name>/ directory)")
    ap.add_argument("--no-dataset-summary", dest="print_dataset_summary",
                    action="store_false", default=True,
                    help="Suppress the dataset summary printout (useful for repeated "
                         "loads of the same data, e.g. multi-seed runs)")

    ap.add_argument("--gnn-name", type=str, default="sage",
                    help="GNN architecture name (for W&B logging only; edit models.py to implement)")

    ap.add_argument("--data-type", type=str, choices=["raw", "clean", "mean", "random"], default="clean",
                    help="Graph variant: 'clean' (default; build_clean_ctx drops alias Metabolite/"
                         "GeneProduct and blank-subtype Interaction nodes), 'raw' (the unmodified "
                         "graph used before the data-cleaning merge), 'mean' or 'random' (clean, "
                         "then impute remaining zero-x rows, see randomize_zero_features)")
    ap.add_argument("--layer-norm", dest="layer_norm", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Per-node-type LayerNorm after each GNN layer (sage/hgt/gat/rgcn); "
                         "on by default, disable with --no-layer-norm.")
    ap.add_argument("--remove-metabolite-organism-edges", dest="remove_metabolite_organism_edges",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Drop (Metabolite, organism, Organism) edges; on by default, keep them "
                         "with --no-remove-metabolite-organism-edges.")
    ap.add_argument("--bidirectional", dest="bidirectional",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Add reverse edges: (Organism, rev_organism, Protein/GeneProduct) "
                         "when Organism nodes are present, and (Metabolite, "
                         "rev_participants, Interaction). On by default, disable with "
                         "--no-bidirectional.")

    return ap.parse_args()


def _train_step(model, optimizer, ctx, data, conv_idxs, neg_k, rng, neg_k_cp, exclude_pairs, grad_clip=0.5):
    model.train()
    optimizer.zero_grad()

    lbl    = data[ctx.target_edge].edge_label
    pos_ei = data[ctx.target_edge].edge_label_index[:, lbl == 1]

    eli, labels = sample_training_negatives(
        pos_ei, conv_idxs, neg_k, rng, ctx.device,
        n_prot=data[ctx.src_type].num_nodes,
        neg_k_cp=neg_k_cp,
        exclude_pairs=exclude_pairs,
    )

    logits = model(ctx, data.x_dict, data.edge_index_dict, eli)
    loss   = F.binary_cross_entropy_with_logits(logits, labels)

    if not torch.isfinite(loss):
        return float("nan"), True

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item(), False


def _evaluate(model, predictor, ctx, args, epoch, exclude_pairs):
    model.eval()
    val_phk = protein_hits_at_k(
        model, predictor, ctx.val_data, ctx,
        k_list=(10, 50), eval_embedded_only=True, any_catalyst_lookup=None,
    )
    val_cp_auc, val_cp_ap = evaluate_cp_auc(
        model, predictor, ctx.val_data, ctx,
        rng=np.random.default_rng(args.seed + epoch), neg_k=1, exclude_pairs=exclude_pairs,
    )
    _, _, val_loss  = evaluate_random_neg_auc(model, predictor, ctx.val_data,  ctx)
    _, _, test_loss = evaluate_random_neg_auc(model, predictor, ctx.test_data, ctx)
    return val_cp_auc, val_cp_ap, val_loss, test_loss, val_phk["P-H@10"], val_phk["P-H@50"]


def run(args, run_dir: Path) -> dict:
    """Train one model and return the final results dict."""
    print(f"Run: {args.run_name}  seed={args.seed}  →  {run_dir}")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}  ({gpu})")
    print()

    ctx = load_ctx_for_run(vars(args), args.data_dir)

    if args.data_type != "raw":
        print(f"\nGraph variant '{args.data_type}':")
        for nt, x in ctx.train_data.x_dict.items():
            n_zero = int((x.abs().sum(dim=1) == 0).sum())
            print(f"  {nt:<12}  num_nodes={x.shape[0]:>7,}  zero-input rows={n_zero:,} "
                  f"({100*n_zero/x.shape[0]:.1f}%)")
        for et in ctx.train_data.edge_types:
            print(f"  {str(et):<50}  num_edges={ctx.train_data[et].num_edges:>10,d}")
        print()

    model = build_model(
        ctx, gnn_name=args.gnn_name,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        decoder=args.decoder, dropout=args.dropout, random_seed=args.seed,
        use_norm=args.layer_norm,
    )

    if not args.use_embeddings:
        print("Replacing pre-computed embeddings with random features (structural baseline)")
        torch.manual_seed(args.seed)
        for split in (ctx.train_data, ctx.val_data, ctx.test_data):
            for nt in split.node_types:
                dim = split[nt].x.shape[-1]
                split[nt].x = torch.randn(split[nt].num_nodes, dim, device=ctx.device)
        # Note: embedded_protein_mask is left intact (not nulled) so evaluation
        # still restricts to the standard 8,445-protein pool used by every
        # other model, keeping this a controlled structure-only comparison
        # rather than also changing the ranking pool size.

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma,
        )
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_gamma, patience=args.lr_patience,
        )

    hparams = {**vars(args), "device": str(ctx.device), "gpu": gpu if torch.cuda.is_available() else "CPU"}

    exclude_pairs = collect_all_pairs(ctx)
    rng           = np.random.default_rng(args.seed)

    if args.neg_pool == "catalyzed":
        train_conv_idxs = list(ctx.all_catalyst_lookup.keys())
        print(f"  neg_pool=catalyzed: sampling (Protein, Conversion_random) negatives from "
              f"{len(train_conv_idxs):,}/{len(ctx.conv_idxs):,} Conversion reactions with a known catalyst")
    else:
        train_conv_idxs = ctx.conv_idxs

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_train_start = time.time()
    run_tag    = Path(args.run_name).name
    best_ph50  = -1.0
    best_epoch = 0
    best_path  = run_dir / f"{run_tag}_best.pt"
    no_improve = 0

    latest_path: Path | None = None

    history: dict = {k: [] for k in [
        "train_loss", "val_ph10", "val_ph50", "val_cp_auc", "val_cp_ap", "val_loss", "test_loss", "lr",
    ]}

    print(f"{'Epoch':>6}  {'loss':>7} {'val_loss':>7} {'tst_loss':>8} "
          f"{'P-H@10':>7}  {'P-H@50':>7}  {'CP-AUC':>7}  {'lr':>9}  {'time':>6}")
    print("─" * 66)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, diverged = _train_step(
            model, optimizer, ctx, ctx.train_data, train_conv_idxs,
            args.neg_k, rng, args.neg_k_cp, exclude_pairs,
            grad_clip=args.grad_clip,
        )
        if diverged:
            print(f"Epoch {epoch}: loss diverged — stopping.")
            break

        val_cp_auc, val_cp_ap, val_loss, test_loss, ph10, ph50 = _evaluate(
            model.gnn, model.predictor, ctx, args, epoch, exclude_pairs,
        )

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        dt = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(loss)
        history["val_ph10"].append(ph10)
        history["val_ph50"].append(ph50)
        history["val_cp_auc"].append(val_cp_auc)
        history["val_cp_ap"].append(val_cp_ap)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)
        history["lr"].append(cur_lr)

        if epoch % args.log_every == 0:
            print(f"{epoch:>6d}  {loss:>7.4f}  {val_loss:>7.4f}  {test_loss:>8.4f} "
                  f"{ph10:>7.4f}  {ph50:>7.4f}  {val_cp_auc:>7.4f}  {cur_lr:>9.2e}  {dt:>5.1f}s")

        if latest_path is not None and latest_path.exists():
            latest_path.unlink()
        latest_path = run_dir / f"{args.gnn_name}_L{args.num_layers}_latest_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch, "loss": loss, "val_ph50": ph50,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "hparams": hparams,
        }, latest_path)

        if ph50 > best_ph50:
            best_ph50  = ph50
            best_epoch = epoch
            no_improve = 0
            if best_path is not None and best_path.exists():
                best_path.unlink()
            best_path = run_dir / f"{args.gnn_name}_L{args.num_layers}_epoch{epoch:03d}_best.pt"
            torch.save(torch.load(latest_path, weights_only=False), best_path)
        else:
            no_improve += 1

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"\nEpoch {epoch}: no P-H@50 improvement in "
                  f"{args.early_stop_patience} epochs — early stop.")
            break

    history_path = run_dir / f"history_{args.gnn_name}_epoch{best_epoch:03d}.json"
    history_path.write_text(json.dumps(history, indent=2))

    print(f"\nBest checkpoint: epoch {best_epoch}  val P-H@50 = {best_ph50:.4f}")
    best_ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])

    print("\n── Final evaluation (best checkpoint) ──────────────────────────────")
    results: dict = {"best_epoch": best_epoch}
    for split_name, data in [("val", ctx.val_data), ("test", ctx.test_data)]:
        phk = protein_hits_at_k(
            model.gnn, model.predictor, data, ctx,
            k_list=(1, 5, 10, 50), eval_embedded_only=True, any_catalyst_lookup=None,
        )
        cp_auc, cp_ap = evaluate_cp_auc(
            model.gnn, model.predictor, data, ctx,
            rng=np.random.default_rng(args.seed + 100), neg_k=1, exclude_pairs=exclude_pairs,
        )
        _, _, split_loss = evaluate_random_neg_auc(model.gnn, model.predictor, data, ctx)
        print(f"  {split_name}:")
        print(f"    P-H@1={phk['P-H@1']:.4f}  P-H@5={phk['P-H@5']:.4f}  "
              f"P-H@10={phk['P-H@10']:.4f}  P-H@50={phk['P-H@50']:.4f}")
        print(f"    CP-AUC={cp_auc:.4f}  CP-AP={cp_ap:.4f}  loss={split_loss:.4f}")
        results.update({
            f"{split_name}_ph50": phk["P-H@50"], f"{split_name}_ph10": phk["P-H@10"],
            f"{split_name}_ph5":  phk["P-H@5"],  f"{split_name}_ph1":  phk["P-H@1"],
            f"{split_name}_cp_auc": cp_auc,       f"{split_name}_cp_ap": cp_ap,
            f"{split_name}_loss":   split_loss,
        })

    total_train_time = time.time() - t_train_start
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    )
    compute_stats = {
        "total_train_time_s": round(total_train_time, 1),
        "device": str(ctx.device),
        "gpu": gpu if torch.cuda.is_available() else "CPU",
        "peak_gpu_memory_mb": round(peak_gpu_mb, 1),
        "epochs_run": best_epoch + (args.early_stop_patience if no_improve >= args.early_stop_patience else args.epochs),
    }


    report_path = run_dir / f"report_{args.gnn_name}_epoch{best_epoch:03d}.json"
    save_report(
        report_path,
        hparams=hparams,
        dataset_stats=collect_dataset_stats(ctx),
        model_stats=collect_model_stats(model.gnn, model.predictor, ctx),
        results=results,
        compute_stats=compute_stats,
    )

    print(f"Checkpoint → {best_path}")
    print(f"Log        → {run_dir / 'train.log'}")
    print(f"History    → {history_path}")
    return results


def main():
    args = get_args()
    set_seed(args.seed)

    config_tag = f"{args.gnn_name}_L{args.num_layers}_h{args.hidden_dim}_{args.decoder}_lr{args.lr}"
    run_dir = RUNS_DIR / args.run_name / config_tag / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with TeeLogger(run_dir / "train.log"):
        run(args, run_dir)


if __name__ == "__main__":
    main()
