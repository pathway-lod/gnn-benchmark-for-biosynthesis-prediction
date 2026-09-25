#!/usr/bin/env python3
"""Evaluation metrics for PlantMetBench.

Primary metric — P-H@50 (Protein Hits at K):
  "For this reaction, is the true catalyst in the model's top-50 ranked proteins?"
  Pool    : embedded proteins only (~8,445)
  Eval on : edges whose true catalyst has an ESM embedding (eval_embedded_only=True)
  Random baseline : 50/8445 ≈ 0.59%

Secondary metric — CP-AUC (Corrupt-Protein AUC):
  "Can the model score the true (Protein, Reaction) pair above a random-protein pair?"
  Random baseline : 0.50

Both metrics focus on the protein-ranking direction (which protein catalyses this
reaction?), which is the primary research question. Do NOT rely on the standard
AUC from evaluate_random_neg_auc() as a model-quality indicator — it inflates
close to 1 because of Pathway co-membership even for random embeddings.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from models import DotPredictor


def _score_all(predictor, pool_embs, query_embs):
    if isinstance(predictor, DotPredictor):
        return pool_embs @ query_embs.T

    n_pool = pool_embs.shape[0]
    n_eval = query_embs.shape[0]
    scores = torch.empty((n_pool, n_eval), device=pool_embs.device)
    pool_ix = torch.arange(n_pool, device=pool_embs.device)

    for j in range(n_eval):
        eli = torch.stack([pool_ix, torch.full_like(pool_ix, j)])  # [2, n_pool]
        scores[:, j] = predictor(pool_embs, query_embs, eli)       # [n_pool]
    return scores


@torch.no_grad()
def protein_hits_at_k(
    model, predictor, data, ctx,
    k_list: tuple[int, ...] = (1, 5, 10, 50),
    eval_embedded_only: bool = True,
    any_catalyst_lookup: dict | None = None,
    no_mp_is_part_of: bool = False,
) -> dict:
    """Protein-ranking H@K (research-question metric).

    For each held-out (reaction, true_catalyst) edge, ranks the true catalyst
    among all proteins in the pool using dot-product scores, then reports the
    fraction of queries where the rank is ≤ K.

    Parameters
    ----------
    model, predictor : the GNN and link-prediction head
    data : one of ctx.val_data or ctx.test_data
    ctx  : GraphContext from dataset.load_data()
    k_list : K values to compute H@K for (default 1, 5, 10, 50)
    eval_embedded_only : skip positive edges whose true catalyst has no embedding
        rather than counting them as guaranteed misses.
        Recommended True — removes embedding-coverage as a confound.
        Requires ctx.embedded_protein_mask to be set (embedded_only_ranking=True
        in load_data, which is the default).
    any_catalyst_lookup : ctx.all_catalyst_lookup — if provided, also computes
        anyP-H@K: a hit if ANY known catalyst for that reaction (across all splits)
        is in top K. Removes false penalties for correctly ranking isoenzymes.
    no_mp_is_part_of : exclude is_part_of edges from message passing

    Returns a dict with keys P-H@{k} for each k, plus pool_size / n_eval / n_total,
    and optionally anyP-H@{k} / any_n_eval.
    """
    model.eval()
    edge_dict = (
        {k: v for k, v in data.edge_index_dict.items() if k[1] != "is_part_of"}
        if no_mp_is_part_of else data.edge_index_dict
    )
    z     = model(data.x_dict, edge_dict)
    z_src = z[ctx.src_type]    # [n_proteins,    dim]
    z_dst = z[ctx.dst_type]    # [n_conversions, dim]

    lbl       = data[ctx.target_edge].edge_label
    pos_ei    = data[ctx.target_edge].edge_label_index[:, lbl == 1]
    n_total   = pos_ei.shape[1]
    if n_total == 0:
        return {f"P-H@{k}": float("nan") for k in k_list}

    # ── Ranking pool ──────────────────────────────────────────────────────────
    prot_mask = ctx.embedded_protein_mask
    if prot_mask is not None:
        prot_mask  = prot_mask.to(z_src.device)
        pool_idx   = torch.where(prot_mask)[0]           # sorted protein indices
        pool_embs  = z_src[pool_idx]
        found      = torch.searchsorted(pool_idx, pos_ei[0])
        in_pool    = (
            (found < pool_idx.shape[0]) &
            (pool_idx[found.clamp(max=pool_idx.shape[0] - 1)] == pos_ei[0])
        )
        pos_src_pool = torch.where(in_pool, found, torch.full_like(found, -1))
    else:
        pool_idx     = torch.arange(z_src.shape[0], device=z_src.device)
        pool_embs    = z_src
        in_pool      = torch.ones(n_total, dtype=torch.bool, device=pos_ei.device)
        pos_src_pool = pos_ei[0]
    n_pool = pool_embs.shape[0]

    # ── Optionally restrict to edges with embedded catalyst ───────────────────
    eval_mask = in_pool if (eval_embedded_only and prot_mask is not None) \
                else torch.ones(n_total, dtype=torch.bool, device=pos_ei.device)
    eval_j    = torch.where(eval_mask)[0]
    n_eval    = eval_j.numel()
    if n_eval == 0:
        result = {f"P-H@{k}": float("nan") for k in k_list}
        result.update({"pool_size": n_pool, "n_eval": 0, "n_total": n_total})
        return result

    pos_ei_eval  = pos_ei[:, eval_j]
    pos_src_eval = pos_src_pool[eval_j]
    in_pool_eval = in_pool[eval_j]

    # ── Vectorised [n_pool × n_eval] scoring ─────────────────────────────────
    dst_embs   = z_dst[pos_ei_eval[1]]
    all_scores = _score_all(predictor, pool_embs, dst_embs) # [n_pool, n_eval]

    all_ranks  = torch.full((n_eval,), n_pool + 1, dtype=torch.long, device=pos_ei.device)
    valid_j    = torch.where(in_pool_eval)[0]
    if valid_j.numel() > 0:
        pool_rows   = pos_src_eval[valid_j]
        # valid_j are LOCAL column indices within all_scores.
        # Using arange(valid_j.numel()) here would be a bug: it reads from the
        # first n_valid columns instead of the correct columns for each edge.
        true_scores = all_scores[pool_rows, valid_j]
        all_ranks[valid_j] = (all_scores[:, valid_j] >= true_scores.unsqueeze(0)).sum(dim=0)

    results = {"pool_size": n_pool, "n_eval": n_eval, "n_total": n_total}
    for k in k_list:
        results[f"P-H@{k}"] = (all_ranks <= k).float().mean().item()

    # ── anyP-H@K ──────────────────────────────────────────────────────────────
    if any_catalyst_lookup is not None:
        pool_pos_map   = {int(pidx): i for i, pidx in enumerate(pool_idx.cpu().tolist())}
        unique_convs   = pos_ei_eval[1].unique()
        uniq_conv_embs = z_dst[unique_convs]
        any_scores = _score_all(predictor, pool_embs, uniq_conv_embs) # [n_pool, n_unique]
        conv_to_col    = {int(c): j for j, c in enumerate(unique_convs.tolist())}

        any_hits: dict[int, list[float]] = {k: [] for k in k_list}
        n_any = 0
        for conv_idx in unique_convs.tolist():
            cats_local = [
                pool_pos_map[p]
                for p in any_catalyst_lookup.get(int(conv_idx), set())
                if p in pool_pos_map
            ]
            if not cats_local:
                if eval_embedded_only:
                    continue
                n_any += 1
                for k in k_list:
                    any_hits[k].append(0.0)
                continue
            n_any += 1
            col      = any_scores[:, conv_to_col[int(conv_idx)]]
            cats_t   = torch.tensor(cats_local, dtype=torch.long, device=col.device)
            max_sc   = col[cats_t].max()
            rank     = int((col >= max_sc).sum())
            for k in k_list:
                any_hits[k].append(float(rank <= k))

        for k in k_list:
            n = len(any_hits[k])
            results[f"anyP-H@{k}"] = sum(any_hits[k]) / n if n > 0 else float("nan")
        results["any_n_eval"] = n_any

    return results


@torch.no_grad()
def evaluate_cp_auc(
    model, predictor, data, ctx,
    rng: np.random.Generator,
    neg_k: int = 1,
    exclude_pairs: set | None = None,
    no_mp_is_part_of: bool = False,
) -> tuple[float, float]:
    """Corrupt-Protein AUC: true (P, C) vs random-protein (P', C).

    For each positive edge, pairs it with neg_k (random Protein, same Conversion)
    pairs, then computes AUC over the combined set. This directly measures whether
    the model can distinguish the true catalyst from random proteins for the same
    reaction — the research question in binary form.

    Returns (auc, average_precision).
    """
    import torch.nn.functional as F
    model.eval()
    edge_dict = (
        {k: v for k, v in data.edge_index_dict.items() if k[1] != "is_part_of"}
        if no_mp_is_part_of else data.edge_index_dict
    )
    z     = model(data.x_dict, edge_dict)
    z_src = z[ctx.src_type]
    z_dst = z[ctx.dst_type]

    lbl    = data[ctx.target_edge].edge_label
    pos_ei = data[ctx.target_edge].edge_label_index[:, lbl == 1].cpu()
    n_prot = data[ctx.src_type].num_nodes
    exclude_pairs = exclude_pairs or set()

    pos_src_np = pos_ei[0].numpy()
    pos_dst_np = pos_ei[1].numpy()
    n = len(pos_src_np)

    # Sample corrupt proteins only from the embedded pool — the same 8,445
    # proteins used for P-H@K ranking.  Zero-embedding proteins are trivially
    # easy negatives that inflate CP-AUC without measuring anything meaningful.
    if ctx.embedded_protein_mask is not None:
        prot_pool_np = torch.where(ctx.embedded_protein_mask.cpu())[0].numpy().astype(np.int64)
    else:
        prot_pool_np = np.arange(n_prot, dtype=np.int64)
    n_prot_pool = len(prot_pool_np)

    neg_src = np.empty((n, neg_k), dtype=np.int64)
    for i in range(n):
        d = int(pos_dst_np[i])
        for j in range(neg_k):
            for _ in range(20):
                cand = int(prot_pool_np[rng.integers(0, n_prot_pool)])
                if (cand, d) not in exclude_pairs:
                    break
            neg_src[i, j] = cand

    neg_src_t = torch.from_numpy(neg_src.ravel()).to(ctx.device)
    neg_dst_t = torch.from_numpy(np.tile(pos_dst_np, neg_k)).to(ctx.device)
    pos_src_t = pos_ei[0].to(ctx.device)
    pos_dst_t = pos_ei[1].to(ctx.device)

    eli    = torch.stack([torch.cat([pos_src_t, neg_src_t]),
                          torch.cat([pos_dst_t, neg_dst_t])])
    labels = torch.cat([torch.ones(n), torch.zeros(n * neg_k)])
    logits = predictor(z_src, z_dst, eli)
    scores = torch.sigmoid(logits).cpu().numpy()
    return (roc_auc_score(labels.numpy(), scores),
            average_precision_score(labels.numpy(), scores))


@torch.no_grad()
def evaluate_random_neg_auc(
    model, predictor, data, ctx,
    no_mp_is_part_of: bool = False,
) -> tuple[float, float, float]:
    """AUC/AP/loss on the 1:1 random-negative pairs in the split.

    WARNING: this metric is unreliable as a training signal for the research
    question. Pathway co-membership inflates it close to 1 even for models that
    have learned nothing protein-specific. Use P-H@50 and CP-AUC instead.
    It is logged here for historical comparability, not as a target metric.

    Returns (auc, average_precision, bce_loss).
    """
    import torch.nn.functional as F
    model.eval()
    edge_dict = (
        {k: v for k, v in data.edge_index_dict.items() if k[1] != "is_part_of"}
        if no_mp_is_part_of else data.edge_index_dict
    )
    z      = model(data.x_dict, edge_dict)
    eli    = data[ctx.target_edge].edge_label_index
    labels = data[ctx.target_edge].edge_label.float()
    logits = predictor(z[ctx.src_type], z[ctx.dst_type], eli)
    loss   = F.binary_cross_entropy_with_logits(logits, labels.to(logits.device)).item()
    scores = torch.sigmoid(logits).cpu().numpy()
    lnp    = labels.cpu().numpy()
    return roc_auc_score(lnp, scores), average_precision_score(lnp, scores), loss
