"""Contrastive objective for full-batch dual-encoder training."""

from __future__ import annotations

import torch
import torch.nn as nn


class FullBatchMLNCELoss(nn.Module):
    r"""Maximum-likelihood noise-contrastive estimation over a full distance matrix.

    Given a distance matrix :math:`d \in \mathbb{R}^{Q \times T}` and the set of
    annotated positive pairs :math:`P`:

    .. math::
        L = \beta \cdot \mathrm{mean}_{(i,j) \in P}\, d_{ij}
            + \log \sum_{i,j} \exp(-\beta \, d_{ij})

    The first term pulls positive pairs together; the log-partition term over *every*
    entry of the matrix pushes all other pairs apart. Unlike InfoNCE this makes no
    one-positive-per-row assumption, so a reaction catalysed by several proteins (and
    a promiscuous protein spanning several reactions) is handled natively.

    Args:
        beta: Inverse temperature. Larger values sharpen the induced distribution.
    """

    def __init__(self, beta: float = 10.0) -> None:
        super().__init__()
        if beta <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta = beta

    def forward(
        self,
        dists: torch.Tensor,
        query_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the scalar loss.

        Args:
            dists: Distance matrix ``[Q, T]``, e.g. ``1 - cosine_similarity``.
            query_idx: Row indices of positive pairs, ``[num_pairs]``, dtype long.
            target_idx: Column indices of positive pairs, ``[num_pairs]``, dtype long.
        """
        if dists.ndim != 2:
            raise ValueError(f"dists must be rank-2, got shape {tuple(dists.shape)}")
        if query_idx.shape != target_idx.shape:
            raise ValueError("query_idx and target_idx must have the same shape")
        if query_idx.numel() == 0:
            raise ValueError("at least one positive pair is required")

        pos_dists = dists[query_idx, target_idx]
        log_partition = torch.logsumexp(-self.beta * dists, dim=(0, 1))
        return self.beta * pos_dists.mean() + log_partition

    def extra_repr(self) -> str:
        return f"beta={self.beta}"
