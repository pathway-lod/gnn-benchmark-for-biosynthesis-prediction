#!/usr/bin/env python3
"""Run logging and report utilities.

Usage in train.py:
    from report import TeeLogger, save_report

    with TeeLogger("runs/myrun/train.log") as log:
        # All print() calls inside this block are saved to train.log AND shown
        # on the terminal — nothing is lost. Exceptions also go to the log.
        ctx = load_data()
        model = build_model(ctx, ...)
        # ... training loop ...
        save_report("runs/myrun/report.json", hparams=hparams, results=results)
"""
from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


# ── TeeLogger ─────────────────────────────────────────────────────────────────

class TeeLogger:
    """Duplicate stdout to both the terminal and a log file.

    Use as a context manager:
        with TeeLogger("train.log"):
            print("this goes to terminal AND train.log")

    Or manually:
        log = TeeLogger("train.log")
        ...
        log.close()
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(path, "w", buffering=1)  # line-buffered
        self._stdout = sys.stdout
        sys.stdout   = self
        self._path   = path
        self.write(f"# Log started at {datetime.now().isoformat()}\n")
        self.write(f"# Log file: {path.resolve()}\n\n")

    def write(self, msg: str) -> None:
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        if sys.stdout is self:
            sys.stdout = self._stdout
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ── Report ────────────────────────────────────────────────────────────────────

def save_report(
    path: str | Path,
    hparams: dict,
    dataset_stats: dict,
    model_stats: dict,
    results: dict,
) -> None:
    """Save a structured JSON report capturing everything about a run.

    Parameters
    ----------
    path         : output path, e.g. "runs/myrun/report.json"
    hparams      : all hyperparameters (from argparse vars(args))
    dataset_stats: graph statistics (node counts, edge counts, split sizes, ...)
    model_stats  : trainable parameter count, edge type count, ...
    results      : final val + test metric values
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp":    datetime.now().isoformat(),
        "git_commit":   _git_commit(),
        "hparams":      hparams,
        "dataset_stats": dataset_stats,
        "model_stats":   model_stats,
        "results":       results,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved → {path.resolve()}")


def collect_dataset_stats(ctx) -> dict:
    """Extract graph statistics from a GraphContext for the report."""
    data = ctx.train_data
    node_counts = {nt: data[nt].num_nodes for nt in sorted(data.node_types)}
    edge_counts = {
        f"{et[0]}__{et[1]}__{et[2]}": data[et].edge_index.shape[1]
        for et in sorted(data.edge_types)
    }
    split_sizes = {name: int(ctx.split_ei[name].shape[1]) for name in ctx.split_ei}
    pool_size   = (
        int(ctx.embedded_protein_mask.sum()) if ctx.embedded_protein_mask is not None
        else node_counts.get("Protein", 0)
    )
    return {
        "node_counts":  node_counts,
        "edge_counts":  edge_counts,
        "split_sizes":  split_sizes,
        "n_edge_types": len(data.edge_types),
        "ranking_pool_size": pool_size,
        "random_ph50":  round(50 / pool_size, 6) if pool_size else None,
    }


def collect_model_stats(model, predictor, ctx) -> dict:
    """Extract model statistics from a built model for the report."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    edge_types_in_gnn = [
        f"{et[0]}__{et[1]}__{et[2]}"
        for et in ctx.train_data.edge_types
        if ctx.train_data[et].edge_index.shape[1] > 0
    ]
    return {
        "n_trainable_params": n_params,
        "n_edge_types_in_gnn": len(edge_types_in_gnn),
        "edge_types_in_gnn":   edge_types_in_gnn,
        "feature_dims": {
            nt: int(ctx.train_data[nt].x.shape[-1])
            for nt in ctx.train_data.node_types
            if hasattr(ctx.train_data[nt], "x") and ctx.train_data[nt].x is not None
        },
    }


def _git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty  = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{commit}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"
