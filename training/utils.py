#!/usr/bin/env python3
"""Training utilities shared across experiments.

Participants generally do NOT need to edit this file.
The main files to edit are:
  models.py  — GNN architecture and decoder
  train.py   — hyperparameters and W&B config
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_all_pairs(ctx) -> set:
    """Return all known positive (protein_idx, conversion_idx) pairs across all splits.

    Used to exclude known true positives when sampling training negatives, so the
    model is never asked to score a real enzyme–reaction pair as a negative example.
    """
    pairs: set = set()
    for d in (ctx.train_data, ctx.val_data, ctx.test_data):
        lbl    = d[ctx.target_edge].edge_label
        pos_ei = d[ctx.target_edge].edge_label_index[:, lbl == 1].cpu()
        pairs.update(zip(pos_ei[0].tolist(), pos_ei[1].tolist()))
    return pairs


def build_pathway_pools(data_dir: Path | str, ctx) -> dict:
    """Build same-pathway protein/conversion pools for hard-negative sampling.

    Loads pathway membership from edges.tsv (the raw edge file, available even
    when is_part_of edges are stripped from the GNN graph). Returns pools that
    map each pathway to the set of Conversion and Protein indices it contains,
    and each Conversion index to the pathways it belongs to.

    These pools can be used to draw negatives from the SAME pathway as each
    positive, forcing the model to learn protein- and reaction-level biochemistry
    rather than exploiting the near-trivial "shares a pathway" shortcut.

    Returns a dict with:
      path_to_conv  : {pathway_id → np.int64 array of Interaction node indices}
      path_to_prot  : {pathway_id → np.int64 array of Protein node indices}
      conv_to_paths : {conv_idx   → list of pathway_id strings}
    """
    edges_df = pd.read_csv(Path(data_dir) / "edges.tsv", sep="\t")

    conv_path_ei = edges_df[
        (edges_df["src_type"] == "Interaction") &
        (edges_df["rel"] == "is_part_of") &
        (edges_df["dst_type"] == "Pathway")
    ]

    inter_rows = ctx.nodes_df[ctx.nodes_df["node_type"] == "Interaction"].reset_index(drop=True)
    inter_id2idx = {nid: i for i, nid in enumerate(inter_rows["node_id"].tolist())}

    conv_to_paths: dict[int, list] = {}
    path_to_conv_raw: dict[str, list] = {}
    for inter_id, path_id in zip(conv_path_ei["src"].tolist(), conv_path_ei["dst"].tolist()):
        c_idx = inter_id2idx.get(str(inter_id))
        if c_idx is None:
            continue
        conv_to_paths.setdefault(c_idx, []).append(str(path_id))
        path_to_conv_raw.setdefault(str(path_id), []).append(c_idx)

    # Protein → Pathway via positives across all splits
    path_to_prot_raw: dict[str, list] = {}
    for split_d in (ctx.train_data, ctx.val_data, ctx.test_data):
        lbl = split_d[ctx.target_edge].edge_label
        pos_ei = split_d[ctx.target_edge].edge_label_index[:, lbl == 1].cpu()
        for p_idx, c_idx in zip(pos_ei[0].tolist(), pos_ei[1].tolist()):
            for path_id in conv_to_paths.get(int(c_idx), []):
                path_to_prot_raw.setdefault(path_id, []).append(int(p_idx))

    n_paths = len(path_to_conv_raw)
    print(f"  [pathway_pools] {n_paths:,} pathways  "
          f"({len(conv_to_paths):,} Conversions with pathway membership)")
    return {
        "conv_to_paths": conv_to_paths,
        "path_to_conv": {p: np.array(sorted(set(v)), dtype=np.int64) for p, v in path_to_conv_raw.items()},
        "path_to_prot": {p: np.array(sorted(set(v)), dtype=np.int64) for p, v in path_to_prot_raw.items()},
    }


def sample_training_negatives(
    pos_ei: torch.Tensor,
    conv_idxs: list,
    neg_k_random: int,
    rng: np.random.Generator,
    device: torch.device,
    n_prot: int = 0,
    neg_k_cp: int = 0,
    exclude_pairs: set | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample training negatives and return (edge_label_index, labels).

    (P, C_random)   neg_k_random per positive
        Same protein, random reaction drawn from the full Conversion pool.
        Trains the model to rank the true reaction above random ones —
        the conversion-ranking direction.

    (P_random, C)   neg_k_cp per positive
        Same reaction, random protein drawn from all proteins.
        Trains the model to rank the true enzyme above random proteins —
        the protein-ranking direction (the research question).
        Requires n_prot > 0.
    """
    pos_src  = pos_ei[0].cpu().numpy()
    pos_dst  = pos_ei[1].cpu().numpy()
    n        = len(pos_src)
    conv_arr = np.asarray(conv_idxs)

    src_list: list[int] = list(pos_src)
    dst_list: list[int] = list(pos_dst)
    lbl_list: list[float] = [1.0] * n

    if neg_k_random > 0:
        for i in range(n):
            s = int(pos_src[i])
            for j in range(neg_k_random):
                for _ in range(20):
                    d = int(conv_arr[rng.integers(0, len(conv_arr))])
                    if exclude_pairs is None or (s, d) not in exclude_pairs:
                        break
                src_list.append(s)
                dst_list.append(d)
                lbl_list.append(0.0)

    if neg_k_cp > 0 and n_prot > 0:
        for i in range(n):
            d = int(pos_dst[i])
            for j in range(neg_k_cp):
                for _ in range(20):
                    cand = int(rng.integers(0, n_prot))
                    if exclude_pairs is None or (cand, d) not in exclude_pairs:
                        break
                src_list.append(cand)
                dst_list.append(d)
                lbl_list.append(0.0)

    eli = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
    lbl = torch.tensor(lbl_list, dtype=torch.float, device=device)
    return eli, lbl

