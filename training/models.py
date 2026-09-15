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
from torch_geometric.nn import HeteroConv, SAGEConv, HGTConv, Linear, GATConv, GraphConv
from torch_geometric.nn import JumpingKnowledge

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

    def __init__(self, edge_types: list, node_types, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3, use_norm: bool = False):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList(
            HeteroConv({et: SAGEConv((-1, -1), hidden_dim) for et in edge_types}, aggr="sum")
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types})
            for _ in range(num_layers)
        ) if use_norm else None

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if self.norms is not None:
                x_dict = {k: self.norms[i][k](v) for k, v in x_dict.items()}
            if i < len(self.convs) - 1:
                x_dict = {
                    k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                    for k, v in x_dict.items()
                }
        return x_dict


class ResidualJumpingHeteroGNN(nn.Module):
    def __init__(self, edge_types: list, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3,
                 jk_mode: str = "cat"):  # "cat" | "max" | "lstm"
        super().__init__()
        self.dropout = dropout
        node_types = sorted({nt for et in edge_types for nt in (et[0], et[2])})

        self.lin_in = nn.ModuleDict({nt: Linear(-1, hidden_dim) for nt in node_types})
        self.norm_in = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types})  # NEW

        self.convs = nn.ModuleList(
            HeteroConv({et: SAGEConv((-1, -1), hidden_dim) for et in edge_types},
                       aggr="mean")  # CHANGED from "sum"
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types})
            for _ in range(num_layers)
        )

        # NEW: per-node-type JK combiner over [layer_0, ..., layer_{L-1}]
        self.jk = nn.ModuleDict({
            nt: JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=num_layers)
            for nt in node_types
        })
        jk_out_dim = hidden_dim * num_layers if jk_mode == "cat" else hidden_dim
        self.lin_out = nn.ModuleDict({  # NEW: project JK output back to hidden_dim
            nt: Linear(jk_out_dim, hidden_dim) for nt in node_types
        })

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        x_dict = {nt: self.norm_in[nt](self.lin_in[nt](x).relu())  # CHANGED: norm added
                  for nt, x in x_dict.items()}

        layer_outputs = {nt: [] for nt in x_dict}  # NEW: collect per-layer states

        for i, conv in enumerate(self.convs):
            h_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                nt: self.norms[i][nt](x_dict[nt] + F.dropout(h, p=self.dropout, training=self.training))
                for nt, h in h_dict.items()
            }
            for nt, x in x_dict.items():  # NEW
                layer_outputs[nt].append(x)
            if i < len(self.convs) - 1:
                x_dict = {nt: F.relu(x) for nt, x in x_dict.items()}

        # NEW: combine multi-hop representations per node type
        out_dict = {
            nt: self.lin_out[nt](self.jk[nt](layer_outputs[nt]))
            for nt in x_dict
        }
        return out_dict


class ResidualHeteroGNN(nn.Module):
    """Stacked HeteroConv layers, one SAGEConv per edge type per layer.

    Design notes:
    - SAGEConv: mean-pool aggregation, handles the wide degree distribution
      (some Proteins are part of hundreds of Pathways)
    - lin_in: per-node-type Linear(-1, hidden_dim) projects the raw features
      (960 Protein, 1024 Metabolite/GeneProduct, 64 random for
      Interaction/Organism/Pathway) to a common hidden_dim up front, so every
      SAGEConv layer is a plain hidden_dim -> hidden_dim block and can carry
      a residual connection
    - residual + LayerNorm: each layer computes x = LN(x + Dropout(conv(x)))
      per node type, which stabilizes deeper stacks and lets gradients skip
      layers directly
    - ReLU only *between* blocks, not after the last one: final embeddings
      feed into a dot product, and clamping them at 0 would prevent the model
      from using negative score directions to push wrong pairs down
    - dropout: standard regularizer, disabled automatically at model.eval()
    """

    def __init__(self, edge_types: list, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        node_types = sorted({nt for et in edge_types for nt in (et[0], et[2])})
        self.lin_in = nn.ModuleDict({nt: Linear(-1, hidden_dim) for nt in node_types})
        self.convs = nn.ModuleList(
            HeteroConv({et: SAGEConv((-1,-1), hidden_dim) for et in edge_types}, aggr="sum")
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types})
            for _ in range(num_layers)
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        x_dict_ = {nt: self.lin_in[nt](x).relu() for nt, x in x_dict.items()} # Attention keeping the same raw residual across all layers

        # this works for 1-layer 
        for i, conv in enumerate(self.convs):
            h_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                nt: self.norms[i][nt](x_dict_[nt] + F.dropout(h, p=self.dropout, training=self.training))
                for nt, h in h_dict.items()
            }
            if i < len(self.convs) - 1:
                x_dict = {nt: F.relu(x) for nt, x in x_dict.items()}
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


class HGT(nn.Module):
    def __init__(self, metadata, hidden=128, num_layers=2, num_heads=2, dropout=0.3, use_norm=False):
        super().__init__()
        node_types, _ = metadata
        self.lin_in = nn.ModuleDict({nt: Linear(-1, hidden) for nt in node_types})
        self.convs = nn.ModuleList(
            HGTConv(hidden, hidden, metadata, num_heads) for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in node_types})
            for _ in range(num_layers)
        ) if use_norm else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {nt: self.lin_in[nt](x).relu() for nt, x in x_dict.items()}
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if self.norms is not None:
                x_dict = {k: self.norms[i][k](v) for k, v in x_dict.items()}
            if i < len(self.convs) - 1:
                x_dict = {nt: self.dropout(x.relu()) for nt, x in x_dict.items()}

        return x_dict



 
class GAT(nn.Module):
    def __init__(self, metadata, hidden=128, num_layers=2, num_heads=2, dropout=0.3, use_norm=False):
        super().__init__()
        node_types, edge_types = metadata
        self.lin_in = nn.ModuleDict({nt: Linear(-1, hidden) for nt in node_types})
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {et: GATConv(hidden, hidden, heads=num_heads, concat=False,
                             add_self_loops=False, dropout=dropout)
                 for et in edge_types},
                aggr="sum")
            self.convs.append(conv)
        self.norms = nn.ModuleList(
            nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in node_types})
            for _ in range(num_layers)
        ) if use_norm else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {nt: self.lin_in[nt](x).relu() for nt, x in x_dict.items()}
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if self.norms is not None:
                x_dict = {k: self.norms[i][k](v) for k, v in x_dict.items()}
            if i < len(self.convs) - 1:
                x_dict = {nt: self.dropout(x.relu()) for nt, x in x_dict.items()}
        return x_dict

class RGCN(nn.Module):
    """Relational GCN — one GraphConv per edge type per layer, summed per node type.

    Ported from rchnn--/rchnn/models.py. Simpler than GAT/HGT (no attention),
    a good baseline to check whether attention is actually earning its keep here.
    """

    def __init__(self, metadata, hidden: int = 128, num_layers: int = 2, dropout: float = 0.3,
                 use_norm: bool = False):
        super().__init__()
        node_types, edge_types = metadata
        self.lin_in = nn.ModuleDict({nt: Linear(-1, hidden) for nt in node_types})
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {et: GraphConv(hidden, hidden) for et in edge_types},
                aggr="sum")
            self.convs.append(conv)
        self.norms = nn.ModuleList(
            nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in node_types})
            for _ in range(num_layers)
        ) if use_norm else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {nt: self.lin_in[nt](x).relu() for nt, x in x_dict.items()}
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if self.norms is not None:
                x_dict = {k: self.norms[i][k](v) for k, v in x_dict.items()}
            if i < len(self.convs) - 1:
                x_dict = {nt: self.dropout(x.relu()) for nt, x in x_dict.items()}
        return x_dict


class MIX(nn.Module):
    """Heterogeneous GNN that mixes SAGEConv and GATConv per edge type.

    Edge types where either endpoint is a target node type (Protein or
    Interaction, i.e. directly involved in the edge being predicted) use
    SAGEConv, on the assumption that all of that neighbor's features are
    relevant. Edge types among the other node types use GATConv, so the
    model learns which of those less directly relevant neighbors to
    attend to.
    """

    def __init__(self, metadata, target_types, hidden: int = 128,
                 num_layers: int = 2, num_heads: int = 2, dropout: float = 0.3):
        super().__init__()
        node_types, edge_types = metadata
        target_types = set(target_types)
        self.lin_in = nn.ModuleDict({nt: Linear(-1, hidden) for nt in node_types})
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for et in edge_types:
                src, _, dst = et
                if src in target_types or dst in target_types:
                    conv_dict[et] = SAGEConv((-1, -1), hidden)
                else:
                    conv_dict[et] = GATConv((-1,-1), hidden, heads=num_heads, concat=False,
                                             add_self_loops=False, dropout=dropout)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {nt: self.lin_in[nt](x).relu() for nt, x in x_dict.items()}
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {nt: self.dropout(x.relu()) for nt, x in x_dict.items()}
        return x_dict


def build_model(
    ctx,
    gnn_name : str = "sage",
    hidden_dim: int = 64,
    num_layers: int = 1,
    decoder: str = "dot",
    dropout: float = 0.3,
    random_seed: int = 42,
    use_norm: bool = False,
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

    node_types = [
        et for et in ctx.train_data.node_types
    ]

    print(f"Node types : {node_types}")
    
    for nt in ctx.train_data.node_types:
        store = ctx.train_data[nt]
        x_shape = tuple(store.x.shape) if "x" in store else None
        print(f"  {nt:<10}  num_nodes={store.num_nodes:>9,d}  x={x_shape}")


    if gnn_name.lower() == "sage":
        print(f" -------- GNN = SAGE ------------")
        gnn = HeteroGNN(edge_types, node_types, hidden_dim=hidden_dim,
                          num_layers=num_layers, dropout=dropout, use_norm=use_norm).to(ctx.device)
    elif gnn_name.lower() == "residual_sage":
        print(f" -------- GNN = Residual SAGE ------------")
        gnn = ResidualHeteroGNN(edge_types, hidden_dim=hidden_dim,
                                   num_layers=num_layers, dropout=dropout).to(ctx.device)
    elif gnn_name.lower() == "residual_jumping_sage":
        print(f" -------- GNN = Residual Jumping SAGE ------------")
        gnn = ResidualJumpingHeteroGNN(edge_types, hidden_dim=hidden_dim,
                                          num_layers=num_layers, dropout=dropout).to(ctx.device)
    elif gnn_name.lower() == "hgt":
        # from hgtconv import HGTConv
        print(f" ---------- GNN = HGT -----------")
        gnn = HGT([node_types, edge_types], hidden=hidden_dim,
                          num_layers=num_layers, dropout=dropout, use_norm=use_norm).to(ctx.device)
    elif gnn_name.lower() == "gat":
        print(f" ----------- GNN = GAT -----------")
        gnn = GAT([node_types, edge_types], hidden=hidden_dim,
                          num_layers=num_layers, dropout=dropout, use_norm=use_norm).to(ctx.device)
    elif gnn_name.lower() == "rgcn":
        print(f" ----------- GNN = RGCN -----------")
        gnn = RGCN([node_types, edge_types], hidden=hidden_dim,
                          num_layers=num_layers, dropout=dropout, use_norm=use_norm).to(ctx.device)
    elif gnn_name.lower() == 'mix':
        print(f" ----------- GNN = MIX -----------")
        gnn = MIX([node_types, edge_types], target_types=(ctx.src_type, ctx.dst_type),
                          hidden=hidden_dim, num_layers=num_layers, dropout=dropout).to(ctx.device)
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
