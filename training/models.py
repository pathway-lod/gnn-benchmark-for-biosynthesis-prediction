#!/usr/bin/env python3
"""GNN models for the PlantMetBench Learnathon.

Baseline (S1): HeteroConv SAGE — one SAGEConv per edge type, two layers, dot-product decoder.
This file is the main one to modify when experimenting with different architectures.

Suggested experiments:
  - Change hidden_dim (default 128) or num_layers (default 2)
  - Replace SAGEConv with GATConv, GINConv, or HGTConv
  - Replace DotPredictor with MLPPredictor (already implemented below)
  - Try different aggregation strategies (aggr="sum" vs "mean" vs "max")
  - Add a residual connection or layer normalization between GNN layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

class HeteroGNN(nn.Module):
    """Stacked HeteroConv layers, one SAGEConv per edge type per layer.

    Design notes:
    - SAGEConv: mean-pool aggregation, handles the wide degree distribution
      (some Proteins are part of hundreds of Pathways)
    - (-1, -1) input dim: lazy init, infers feature dims on first forward()
      (needed since node types have different embedding dims: 960 Protein,
      1024 Metabolite/GeneProduct, 64 random for Interaction/Organism/Pathway)
    - ReLU only *between* layers, not after the last layer: final embeddings
      feed into a dot product, and clamping them at 0 would prevent the model
      from using negative score directions to push wrong pairs down
    - dropout: standard regularizer, disabled automatically at model.eval()
    """

    def __init__(self, edge_types: list, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList(
            HeteroConv({et: SAGEConv((-1, -1), hidden_dim) for et in edge_types}, aggr="sum")
            for _ in range(num_layers)
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {
                    k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                    for k, v in x_dict.items()
                }
        return x_dict


class DotPredictor(nn.Module):
    """Score an edge (src, dst) as z_src · z_dst.

    Simple, parameter-free, and surprisingly effective for link prediction
    when the GNN has already learned discriminative embeddings.
    """

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor,
                edge_label_index: torch.Tensor) -> torch.Tensor:
        src = z_src[edge_label_index[0]]
        dst = z_dst[edge_label_index[1]]
        return (src * dst).sum(dim=-1)


class MLPPredictor(nn.Module):
    """Score an edge via a two-layer MLP on the concatenated embeddings.

    More expressive than the dot product but also more parameters.
    Worth trying if the dot product plateaus early.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor,
                edge_label_index: torch.Tensor) -> torch.Tensor:
        src = z_src[edge_label_index[0]]
        dst = z_dst[edge_label_index[1]]
        return self.mlp(torch.cat([src, dst], dim=-1)).squeeze(-1)


class ModelWithPredictor(nn.Module):
    """Wrap a GNN and a predictor into one module."""
    def __init__(self, gnn: nn.Module, predictor: nn.Module):
        super().__init__()
        self.gnn = gnn
        self.predictor = predictor

    def forward(self, ctx, x_dict:dict, edge_index_dict:dict, edge_label_index: torch.Tensor) -> torch.Tensor:
        z_dict = self.gnn(x_dict, edge_index_dict)
        src = z_dict[ctx.src_type]
        dst = z_dict[ctx.dst_type]
        return self.predictor(src, dst, edge_label_index)





def build_model(
    ctx,
    hidden_dim: int = 128,
    num_layers: int = 2,
    decoder: str = "dot",
    dropout: float = 0.3,
    random_seed: int = 42,
):
    """Instantiate GNN + decoder and run a dummy forward pass to materialize weights.

    SAGEConv(-1, -1) uses lazy parameter initialization: weight shapes are not
    allocated until the first forward() call. The dummy forward here means that
    model.parameters() and load_state_dict() work correctly immediately after
    build_model() returns, without requiring an extra forward pass from the caller.

    Returns: ModelWithPredictor (wraps gnn + predictor), moved to ctx.device
    """
    torch.manual_seed(random_seed)
    edge_types = [
        et for et in ctx.train_data.edge_types
        if ctx.train_data[et].edge_index.shape[1] > 0
    ]
    gnn = HeteroGNN(edge_types, hidden_dim=hidden_dim,
                      num_layers=num_layers, dropout=dropout).to(ctx.device)
    predictor = (
        MLPPredictor(hidden_dim) if decoder == "mlp" else DotPredictor()
    ).to(ctx.device)

    model = ModelWithPredictor(gnn, predictor).to(ctx.device)

    with torch.no_grad():
        print(f'Running dummy forward pass to materialize weights...')
        model.gnn(ctx.train_data.x_dict, ctx.train_data.edge_index_dict)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: HeteroGNN  hidden_dim={hidden_dim}  num_layers={num_layers}  "
          f"decoder={decoder}  dropout={dropout}")
    print(f"  Edge types in GNN: {len(edge_types)}")
    print(f"  Trainable parameters: {n_params:,}")

    return model
