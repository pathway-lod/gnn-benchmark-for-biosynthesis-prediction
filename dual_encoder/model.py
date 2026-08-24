"""Dual-encoder architecture for reaction to protein retrieval.

Two independent MLP towers project reaction fingerprints and protein language-model
embeddings into a shared, L2-normalised embedding space. Because both towers emit
unit-norm vectors, their dot product is a cosine similarity and can be used directly
to rank a protein pool against a query reaction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """Feed-forward tower emitting L2-normalised embeddings.

    Each hidden block is ``Linear -> ReLU -> LayerNorm -> Dropout``, followed by a
    final linear projection to ``output_dim``. Setting ``num_layers=0`` reduces the
    tower to a single linear projection.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 2048,
        num_layers: int = 2,
        dropout: float = 0.3,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError(f"num_layers must be >= 0, got {num_layers}")
        if hidden_dim <= 0 and num_layers > 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU()]
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` of shape ``[N, input_dim]`` into unit-norm ``[N, output_dim]``."""
        return F.normalize(self.net(x), p=2.0, dim=-1)


class DualEncoder(nn.Module):
    """Query (reaction) and target (protein) towers sharing one embedding space.

    The two towers see different input modalities and share no weights; only the
    output dimensionality and the normalisation constraint are common.
    """

    def __init__(
        self,
        query_dim: int,
        target_dim: int,
        emb_dim: int = 512,
        hidden_dim: int = 2048,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        tower_kwargs = dict(
            output_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.query_encoder = MLPEncoder(input_dim=query_dim, **tower_kwargs)
        self.target_encoder = MLPEncoder(input_dim=target_dim, **tower_kwargs)

    def forward(
        self, query_x: torch.Tensor, target_x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of queries and a batch of targets independently."""
        return self.query_encoder(query_x), self.target_encoder(target_x)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def cosine_distances(query_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine distance matrix ``[Q, T]`` for unit-norm inputs."""
    return 1.0 - query_emb @ target_emb.t()
