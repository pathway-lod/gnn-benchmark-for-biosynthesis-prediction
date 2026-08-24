"""Ranking metrics for reaction to protein retrieval.

Two complementary views of the same score matrix:

* **P-H@K** (pool hits@K) -- fraction of positive pairs whose true protein ranks in
  the top K of the *entire* protein pool. This is the hard, pool-sized task.
* **CP-AUC / CP-AP** (contrastive-pair) -- each positive is scored against
  ``num_negatives`` proteins sampled uniformly without replacement from the pool.
  Cheaper, less pool-size dependent, and comparable across benchmark variants.

Ties are resolved pessimistically throughout: a tied candidate counts as ranked above
the positive. With a single positive per query, the closed forms below are exactly
equal to ``sklearn.metrics.roc_auc_score`` and ``average_precision_score``.
"""

from __future__ import annotations

import torch

DEFAULT_KS = (1, 5, 10, 50)


@torch.no_grad()
def retrieval_metrics(
    query_emb: torch.Tensor,
    target_emb: torch.Tensor,
    pairs: torch.Tensor,
    ks: tuple[int, ...] = DEFAULT_KS,
    num_negatives: int = 50,
    seed: int = 0,
    chunk_size: int = 1024,
) -> dict[str, float]:
    """Evaluate positive pairs against the full target pool.

    Args:
        query_emb: Unit-norm query embeddings ``[Q, D]``.
        target_emb: Unit-norm target embeddings ``[P, D]``; the retrieval pool.
        pairs: ``LongTensor[2, N]`` of positive (query index, target index) pairs.
        ks: Cut-offs reported as ``P-H@K``.
        num_negatives: Pool samples per positive for the CP metrics.
        seed: Seed for negative sampling; fixed across runs so the protocol is
            identical for every model and every training seed.
        chunk_size: Positives scored per block, bounding peak memory at
            ``chunk_size x P``.

    Returns:
        Mapping with ``P-H@K`` for each K plus ``cp_auc``, ``cp_ap``, ``pool_size``
        and ``num_pairs``.
    """
    query_emb, target_emb, pairs = query_emb.cpu(), target_emb.cpu(), pairs.cpu()
    pool_size = target_emb.shape[0]
    num_pairs = pairs.shape[1]
    if num_pairs == 0:
        raise ValueError("no positive pairs to evaluate")
    num_negatives = min(num_negatives, pool_size - 1)

    generator = torch.Generator().manual_seed(seed)
    ranks, cp_aucs, cp_aps = [], [], []

    for start in range(0, num_pairs, chunk_size):
        query_idx = pairs[0, start : start + chunk_size]
        target_idx = pairs[1, start : start + chunk_size]
        rows = torch.arange(query_idx.shape[0])

        # Cosine similarity of each positive's query against the whole pool.
        scores = query_emb[query_idx] @ target_emb.t()  # [B, P]
        positive = scores[rows, target_idx]  # [B]
        ranks.append((scores >= positive[:, None]).sum(dim=1))

        # Sample negatives without replacement by ranking random keys, with the
        # true target excluded via a key that can never enter the top-k.
        keys = torch.rand(scores.shape, generator=generator)
        keys[rows, target_idx] = -1.0
        negative = scores.gather(1, keys.topk(num_negatives, dim=1).indices)  # [B, k]

        below = (negative < positive[:, None]).sum(dim=1)
        tied = (negative == positive[:, None]).sum(dim=1)
        cp_aucs.append((below + 0.5 * tied) / num_negatives)
        cp_aps.append(1.0 / (num_negatives - below + 1).float())

    ranks = torch.cat(ranks)
    results: dict[str, float] = {"pool_size": pool_size, "num_pairs": num_pairs}
    for k in ks:
        results[f"P-H@{k}"] = (ranks <= k).float().mean().item()
    results["cp_auc"] = torch.cat(cp_aucs).mean().item()
    results["cp_ap"] = torch.cat(cp_aps).mean().item()
    return results


@torch.no_grad()
def encode_all(
    encoder: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Run ``encoder`` over ``features`` in batches, returning CPU embeddings."""
    encoder.eval()
    return torch.cat(
        [encoder(features[i : i + batch_size].to(device)).cpu() for i in range(0, len(features), batch_size)]
    )
