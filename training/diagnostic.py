import torch
import torch.nn.functional as F
import numpy as np
import time
import json
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11.5,
    "axes.edgecolor": "#444444",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    })

NODE_COLOR = {
        "Protein": "#4C72B0", "GeneProduct": "#DD8452",
        "Interaction": "#55A868", "Metabolite": "#C44E52",
    }

from pathlib import Path
from metrics import evaluate_cp_auc, evaluate_random_neg_auc, protein_hits_at_k
from build_clean_ctx import build_clean_ctx

RUNS_DIR = Path(__file__).parent.parent / "runs_clean"
PLOTS_DIR = Path(__file__).parent / "plots_clean"




@torch.no_grad()
def dirichlet_energy(x_src, x_dst, edge_index):
    # Mean squared distance between connected nodes
    if edge_index.shape[1] == 0:
        return float("nan")

    s,d = edge_index
    return np.round((x_src[s] - x_dst[d]).pow(2).sum(-1).mean().item(),2)


@torch.no_grad()
def mean_avg_distance(x, n_pairs = 2000):
    n = x.shape[0]
    if n<2:
        return float("nan")

    i1 = torch.randint(0,n,(n_pairs,))
    i2 = torch.randint(0,n,(n_pairs,))
    xn = F.normalize(x, dim =-1)
    return (1-(xn[i1]*xn[i2]).sum(-1)).mean().item()



@torch.no_grad()
def mad_neighbor(activations_at_layer, edge_index_dict, node_types):
    """MAD_neighbor per node type: mean cosine distance over actual edges
    touching that node type, pooling both endpoint roles."""
    sums = {nt: 0.0 for nt in node_types}
    counts = {nt: 0 for nt in node_types}
    for (s_t, _, d_t), ei in edge_index_dict.items():
        if ei.shape[1] == 0:
            continue
        s, d = ei
        xs = F.normalize(activations_at_layer[s_t][s], dim=-1)
        xd = F.normalize(activations_at_layer[d_t][d], dim=-1)
        dist = 1 - (xs * xd).sum(-1)          # per-edge cosine distance
        sums[s_t] += dist.sum().item();  counts[s_t] += dist.numel()
        sums[d_t] += dist.sum().item();  counts[d_t] += dist.numel()
    return {nt: (sums[nt] / counts[nt] if counts[nt] > 0 else float("nan"))
            for nt in node_types}



@torch.no_grad()
def effective_rank(x, eps=1e-3):
    """Number of singular values above eps*sigma_max. Oversmoothed embeddings
    collapse onto a low-dim subspace -> rank drops toward 1."""
    if x.shape[0] < 2:
        return 0
    s = torch.linalg.svdvals(x.float())
    return int((s > eps * s[0]).sum().item())


def check_oversmoothing(model, data):
    """model: ModelWithPredictor. data: e.g. ctx.train_data."""
    activations = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            if layer_idx == 0:
                # inp is a tuple of positional args; inp[0] is the x_dict going in
                activations[0] = {nt: x.detach() for nt, x in inp[0].items()}
            activations[layer_idx] = {nt: x.detach() for nt, x in out.items()}
        return hook

    for i, conv in enumerate(model.gnn.convs):
        hooks.append(conv.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        model.gnn(data.x_dict, data.edge_index_dict)
    for h in hooks:
        h.remove()

    edge_index_dict = data.edge_index_dict
    node_types = list(activations[sorted(activations)[0]].keys())
    print(f"{'Layer':>5} {'EdgeType':<40} {'Dirichlet':>10}")
    for i in sorted(activations):
        for et, ei in edge_index_dict.items():
            s_t, _, d_t = et
            if ei.shape[1] == 0:
                continue
            de = dirichlet_energy(activations[i][s_t], activations[i][d_t], ei)
            print(f"{i:>5} {str(et):<40} {de:>10.2f}")

    print(f"\n{'Layer':>5} {'NodeType':<12} {'MAD':>8} {'EffRank':>8} {'MeanNorm':>10}")
    for i in sorted(activations):
        mad_nb = mad_neighbor(activations[i], edge_index_dict, node_types)  
        for nt, x in activations[i].items():
            mad_remote = mean_avg_distance(x)
            madgap = mad_remote - mad_nb[nt]
            print(f"{i:>5} {nt:<12} {mad_remote:>8.4f} {madgap:>8.4f} "
              f"{effective_rank(x):>8d} {x.norm(dim=-1).mean().item():>10.4f}")

    return activations


#### ----------------------------------------- VANISHING GRADIENT CHECK --------------------------------- ###

def check_gradient_vanishing(model, ctx, data, edge_label_index, labels):
    """Run one real backward pass, inspect grad norms per conv layer.
    If layer 0's grad norm is orders of magnitude below the last layer's,
    that's the classic vanishing-gradient signature."""
    model.train()
    model.zero_grad()
    logits = model(ctx, data.x_dict, data.edge_index_dict, edge_label_index)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    print(f"{'Layer':>5} {'#Params':>8} {'GradNorm':>12} {'WeightNorm':>12} {'Grad/Weight':>12}")
    for i, conv in enumerate(model.gnn.convs):
        g2 = w2 = 0.0
        n = 0
        for p in conv.parameters():
            if p.grad is not None:
                g2 += p.grad.pow(2).sum().item()
                w2 += p.pow(2).sum().item()
                n += 1
        g, w = g2 ** 0.5, w2 ** 0.5
        print(f"{i:>5} {n:>8} {g:>12.3e} {w:>12.3e} {g/(w+1e-12):>12.3e}")


def check_input_gradient_flow(model, ctx, data, edge_label_index, labels):
    """Complementary check: does gradient signal actually reach the input
    features at all, or does it die before layer 0?"""
    model.train()
    model.zero_grad()
    x_dict = {k: v.clone().requires_grad_(True) for k, v in data.x_dict.items()}
    logits = model(ctx, x_dict, data.edge_index_dict, edge_label_index)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    print("Input-feature gradient norms:")
    for nt, x in x_dict.items():
        g = x.grad.norm().item() if x.grad is not None else 0.0
        print(f"  {nt:<12} {g:.6e}")



#### ----------------------------------------- PLOT --------------------------------- ###

def style_ax(ax, ylabel, title, layers):
    ax.set_title(title, pad = 10)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_xticks(layers)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)

def plot_node_metrics(activations, edge_index_dict, save_path=PLOTS_DIR, subtitles=""):
    layers = sorted(activations)
    node_types = list(activations[layers[0]].keys())
    mad_nb_per_layer = {i: mad_neighbor(activations[i], edge_index_dict, node_types) for i in layers}

    metrics = {
        nt: {
            "mad":       [mean_avg_distance(activations[i][nt]) for i in layers],
            "madgap":    [mean_avg_distance(activations[i][nt]) - mad_nb_per_layer[i][nt] for i in layers],
            "eff_rank":  [effective_rank(activations[i][nt]) for i in layers],
            "mean_norm": [activations[i][nt].norm(dim=-1).mean().item() for i in layers],
        } for nt in node_types
    }
    full_rank = activations[layers[0]][next(iter(node_types))].shape[-1]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for nt, m in metrics.items():
        c = NODE_COLOR.get(nt, "#888888")
        for ax, key in zip(axes, ["mad", "madgap", "eff_rank", "mean_norm"]):
            ax.plot(layers, m[key], marker="o", markersize=6.5, linewidth=2.4,
                    color=c, markeredgecolor="white", markeredgewidth=1, zorder=3)
        axes[3].annotate(nt, xy=(layers[-1], m["mean_norm"][-1]), xytext=(6, 0),
                          textcoords="offset points", va="center", fontsize=9.5,
                          color=c, fontweight="bold")

    axes[1].axhline(0, color="#999999", linestyle="--", linewidth=1.2, zorder=1)
    axes[2].axhline(full_rank, color="#999999", linestyle="--", linewidth=1.2, zorder=1)
    axes[2].annotate(f"full rank ({full_rank})", xy=(layers[0], full_rank),
                      xytext=(0, 4), textcoords="offset points",
                      fontsize=9, color="#999999", style="italic")

    style_ax(axes[0], "mean cosine distance", "MAD (random pairs)", layers)
    style_ax(axes[1], "MAD_remote − MAD_neighbor", "MADGap\n(higher = healthier)", layers)
    style_ax(axes[2], "# singular values > ε·σₘₐₓ", "Effective rank", layers)
    style_ax(axes[3], "mean ‖z‖", "Mean embedding norm", layers)
    axes[3].set_xlim(right=layers[-1] + 0.9)

    fig.tight_layout(rect=(0, 0, 1, 0.78))
    #fig.text(0.5, 0.90, "Oversmoothing diagnostics per node type",
    #          ha="center", fontsize=15, fontweight="bold")
    if subtitles:
        fig.text(0.5, 0.855, subtitles, ha="center", fontsize=10, color="#666666", style="italic")
    handles = [plt.Line2D([0], [0], color=NODE_COLOR.get(nt, "#888888"), lw=2.4,
                           marker="o", markersize=6.5) for nt in metrics]
    fig.legend(handles, metrics.keys(), loc="center", ncol=len(metrics),
               frameon=False, bbox_to_anchor=(0.5, 0.815), fontsize=11)
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    print(f"Saved -> {save_path}")
    return fig


def plot_dirichlet_energy(activations, edge_index_dict, target_edge=None,
                           save_path=PLOTS_DIR / "dirichlet_energy.png", subtitles=""):
    layers = sorted(activations)
    per_edge = {}
    for et, ei in edge_index_dict.items():
        if ei.shape[1] == 0:
            continue
        s_t, _, d_t = et
        per_edge[et] = [dirichlet_energy(activations[i][s_t], activations[i][d_t], ei) for i in layers]

    palette = plt.cm.tab20(range(len(per_edge)))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for (et, vals), c in zip(per_edge.items(), palette):
        if et == target_edge:
            ax.plot(layers, vals, marker="o", markersize=7, linewidth=3.2, color="#C44E52",
                    zorder=5, label=f"{et} — task edge", markeredgecolor="white", markeredgewidth=1)
        else:
            ax.plot(layers, vals, marker="o", markersize=5, linewidth=2.0, color=c,
                    zorder=2, label=str(et))
    ax.set_yscale("log")
    style_ax(ax, "Dirichlet energy (log scale)", "", layers)
    ax.legend(fontsize=8.3, loc="upper left", bbox_to_anchor=(1.02, 1.02),
              frameon=False, labelspacing=0.6)

    fig.tight_layout(rect=(0, 0, 0.98, 0.80))
    fig.text(0.5, 0.87, "Dirichlet energy per edge type across layers",
              ha="center", fontsize=15, fontweight="bold")
    if subtitles:
        fig.text(0.5, 0.835, subtitles, ha="center", fontsize=10, color="#666666", style="italic")
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    print(f"Saved -> {save_path}")
    return fig


def plot_gradient_vanishing(model, ctx, data, edge_label_index, labels,
                             save_path=PLOTS_DIR / "gradient_vanishing.png", subtitles=""):
    """Run one backward pass on the model as-is (trained or not) and plot
    grad norm / grad-to-weight ratio per conv layer. Look at the RATIO panel —
    if layer 0's ratio is orders of magnitude below the last layer's, that's
    the vanishing-gradient signature, not just 'gradients are small overall'."""
    model.train()
    model.zero_grad()
    logits = model(ctx, data.x_dict, data.edge_index_dict, edge_label_index)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    layers = list(range(len(model.gnn.convs)))
    grad_norms, weight_norms, ratios = [], [], []
    for conv in model.gnn.convs:
        g2 = w2 = 0.0
        for p in conv.parameters():
            if p.grad is not None:
                g2 += p.grad.pow(2).sum().item()
                w2 += p.pow(2).sum().item()
        g, w = g2 ** 0.5, w2 ** 0.5
        grad_norms.append(g); weight_norms.append(w); ratios.append(g / (w + 1e-12))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(layers, grad_norms, marker="o", markersize=7, linewidth=2.4,
                 color="#C44E52", markeredgecolor="white", markeredgewidth=1)
    axes[0].set_yscale("log")
    style_ax(axes[0], "‖grad‖ (log scale)", "Parameter gradient norm per layer", layers)

    axes[1].plot(layers, ratios, marker="o", markersize=7, linewidth=2.4,
                 color="#DD8452", markeredgecolor="white", markeredgewidth=1)
    axes[1].set_yscale("log")
    style_ax(axes[1], "grad / weight (log scale)", "Grad-to-weight ratio per layer", layers)

    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.text(0.5, 0.90, "Vanishing gradients", ha="center", fontsize=15, fontweight="bold")
    if subtitles:
        fig.text(0.5, 0.855, subtitles, ha="center", fontsize=10, color="#666666", style="italic")
    fig.savefig(save_path, dpi=170)
    print(f"Saved -> {save_path}")
    return fig, {"grad_norm": grad_norms, "weight_norm": weight_norms, "ratio": ratios}


def plot_input_gradient_flow(model, ctx, data, edge_label_index, labels,
                              save_path=PLOTS_DIR / "input_gradient_flow.png"):
    """Two-panel horizontal bar chart, per node type:
    (a) relative reliance -- share of the total input-feature gradient, i.e.
        how much each entity contributes to the final decision;
    (b) absolute magnitude -- raw grad norm on a log scale, i.e. whether
        signal reaches that node type's inputs at all."""


    node_types = list(data.x_dict.keys())
    for nt in node_types:
        x = data.x_dict[nt].detach()
        n_zero = (x.abs().sum(dim=1) == 0).sum().item()
        print(f"{nt}: {n_zero}/{x.shape[0]} zero-input rows ({100*n_zero/x.shape[0]:.1f}%)")


    model.train()
    model.zero_grad()
    x_dict = {k: v.clone().requires_grad_(True) for k, v in data.x_dict.items()}
    logits = model(ctx, x_dict, data.edge_index_dict, edge_label_index)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    # RMS per-element grad magnitude, not the raw L2 norm: a type with more
    # nodes or a higher-dim input embedding would otherwise look more
    # "important" purely from having a bigger tensor, not a stronger signal
    node_types = list(x_dict.keys())
    grad_norms = {nt: (x_dict[nt].grad.norm(dim=1).mean().item() if x_dict[nt].grad is not None else 0.0)
              for nt in node_types}
    order = sorted(node_types, key=lambda nt: grad_norms[nt], reverse=True)
    values = [grad_norms[nt] for nt in order]
    colors = [NODE_COLOR.get(nt, "#888888") for nt in order]

    total = sum(values) or 1.0
    shares = [100 * v / total for v in values]


    node_types = list(x_dict.keys())
    for nt in node_types:
        x = x_dict[nt].detach()
        n_zero = (x.abs().sum(dim=1) == 0).sum().item()
        print(f"{nt}: {n_zero}/{x.shape[0]} zero-input rows ({100*n_zero/x.shape[0]:.1f}%)")


    # log scale can't show true zeros, so floor them to a visible sliver;
    # the label still shows the real (possibly zero) value, and the bar
    # itself is hatched/faded so a floored zero can't be misread as a
    # small-but-real magnitude next to genuine small values
    nonzero = [v for v in values if v > 0]
    floor = (min(nonzero) / 20) if nonzero else 1e-8
    log_heights = [v if v > 0 else floor for v in values]
    is_floored = [v == 0 for v in values]

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.85 * len(order) + 0.9))

    ax = axes[0]
    bars = ax.barh(order, shares, color=colors, edgecolor="white", linewidth=1.2,
                    height=0.6, zorder=3)
    for bar, s in zip(bars, shares):
        ax.text(bar.get_width() + max(shares) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{s:.1f}%", va="center", fontsize=9.5, color="#333333")
    ax.set_xlim(0, max(shares) * 1.35)
    # ax.set_xlabel("Share of total input-feature gradient (%)")
    #ax.set_title("(a) Relative reliance", loc="left")

    ax = axes[1]
    bars = ax.barh(order, log_heights, color=colors, edgecolor="white", linewidth=1.2,
                    height=0.6, zorder=3)
    for bar, floored in zip(bars, is_floored):
        if floored:
            bar.set_alpha(0.35)
            bar.set_hatch("///")
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height() / 2,
                 "0" if v == 0 else f"{v:.2g}", va="center", fontsize=9.5, color="#333333")
    ax.set_xscale("log")
    ax.set_xlim(floor * 0.2, max(log_heights) * 20)
    # ax.set_xlabel(r"Input-feature grad norm $\Vert\partial\mathcal{L}/\partial x_t\Vert$ (log)")
   #  ax.set_title("(b) Absolute magnitude", loc="left")

    for ax in axes:
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    print(f"Saved -> {save_path}")
    return fig

    

# ============================================================
# Grouped: large-magnitude vs small-magnitude, legends below each panel
# ============================================================
def plot_dirichlet_energy_grouped(activations, edge_index_dict, target_edge=None,
                                   split_threshold=10.0,
                                   save_path=PLOTS_DIR / "dirichlet_energy_grouped.png",
                                   subtitles=""):
    layers = sorted(activations)
    per_edge = {}
    for et, ei in edge_index_dict.items():
        if ei.shape[1] == 0:
            continue
        s_t, _, d_t = et
        per_edge[et] = [dirichlet_energy(activations[i][s_t], activations[i][d_t], ei) for i in layers]

    large = {et: v for et, v in per_edge.items() if v[0] >= split_threshold}
    small = {et: v for et, v in per_edge.items() if v[0] < split_threshold}
    distinct_colors = ["#1b9e77", "#7570b3", "#e6ab02", "#66a61e",
                        "#a6761d", "#e7298a", "#377eb8"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    ax = axes[0]
    gi = 0
    for et, vals in large.items():
        if et == target_edge:
            ax.plot(layers, vals, marker="o", markersize=7, linewidth=3.2, color="#C44E52",
                    zorder=5, label=f"{et} — task edge", markeredgecolor="white", markeredgewidth=1)
        else:
            ax.plot(layers, vals, marker="o", markersize=5, linewidth=2.0,
                    color=distinct_colors[gi % len(distinct_colors)], zorder=2, label=str(et))
            gi += 1
    ax.set_yscale("log")
    style_ax(ax, "Dirichlet energy (log scale)",
             "Large-magnitude edges\n(sharp early drop, then plateau)", layers)
    ax.legend(fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              frameon=False, labelspacing=0.5, ncol=2)

    ax = axes[1]
    small_palette = plt.cm.tab10(range(len(small)))
    for (et, vals), c in zip(small.items(), small_palette):
        ax.plot(layers, vals, marker="o", markersize=6.5, linewidth=2.4, color=c,
                markeredgecolor="white", markeredgewidth=1, label=str(et))
    style_ax(ax, "Dirichlet energy", "Small-magnitude edges\n(check for norm-growth confound)", layers)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              frameon=False, labelspacing=0.6, ncol=1)

    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    fig.text(0.5, 0.975, "Dirichlet energy per edge type across layers",
              ha="center", fontsize=15, fontweight="bold")
    if subtitles:
        fig.text(0.5, 0.945, subtitles, ha="center", fontsize=10, color="#666666", style="italic")
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    print(f"Saved -> {save_path}")
    return fig


def run(args, run_dir: Path) -> dict:
    """Train one model and return the final results dict."""
    print(f"Run: {args.run_name}  seed={args.seed}  →  {run_dir}")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}  ({gpu})")
    print()

    ctx = load_data(
        data_dir=args.data_dir,
        random_seed=args.seed,
        remove_is_part_of=True,
        embedded_only_ranking=True,
        disjoint_train_ratio=0.2,
        keep_catalyzed_by=False,
        load_ec_embeddings=args.ec_features,
        remove_currency_metabolites=args.remove_currency_metabolites,
        remove_all_metabolites=args.remove_all_metabolites,
        organism_embeddings_path=args.organism_embeddings_path,
        organism_embedding_type=args.organism_embedding_type,
        remove_gene_organism_edges=args.remove_gene_organism_edges,
        remove_organism=args.remove_organism,
        download=args.download,
        print_summary=args.print_dataset_summary,
    )

    ctx = build_clean_ctx(ctx)    
    print(f"\n -------- CLEAN DATA --------- \n")
    print("Node types:")
    for nt in ctx.train_data.node_types:
        store = ctx.train_data[nt]
        x_shape = tuple(store.x.shape) if "x" in store else None
        print(f"  {nt:<12}  num_nodes={store.num_nodes:>9,d}  x={x_shape}")
    print("Edge types:")
    for et in ctx.train_data.edge_types:
        store = ctx.train_data[et]
        print(f"  {str(et):<50}  num_edges={store.num_edges:>10,d}")
    print()


    model = build_model(
        ctx, gnn_name = args.gnn_name, 
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        decoder=args.decoder, dropout=args.dropout, random_seed=args.seed,
    )

    if any(run_dir.glob(f"{args.gnn_name}_L{args.num_layers}_latest_epoch*.pt")):
        p = sorted(run_dir.glob(f"{args.gnn_name}_L{args.num_layers}_latest_epoch*.pt"))[-1]
        best_ckpt_ = torch.load(p, weights_only=False)
        model.load_state_dict(best_ckpt_["model_state"])
        print(f"Loaded model from {p} (epoch {best_ckpt_['epoch']})")
        return model, ctx
        

    if not args.use_embeddings:
        print("Replacing pre-computed embeddings with random features (structural baseline)")
        torch.manual_seed(args.seed)
        for split in (ctx.train_data, ctx.val_data, ctx.test_data):
            for nt in split.node_types:
                dim = split[nt].x.shape[-1]
                split[nt].x = torch.randn(split[nt].num_nodes, dim, device=ctx.device)
        ctx.embedded_protein_mask = None

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    hparams = {**vars(args), "device": str(ctx.device), "gpu": gpu if torch.cuda.is_available() else "CPU"}

    exclude_pairs = collect_all_pairs(ctx)
    rng           = np.random.default_rng(args.seed)

    latest_path: Path | None = None
    best_ph50   = -1.0
    best_epoch  = 0
    best_path: Path | None = None
    no_improve  = 0

    history: dict = {k: [] for k in [
        "train_loss", "val_ph10", "val_ph50", "val_cp_auc", "val_cp_ap", "val_loss", "test_loss",
    ]}

    print(f"{'Epoch':>6}  {'loss':>7} {'val_loss':>7} {'tst_loss':>8} "
          f"{'P-H@10':>7}  {'P-H@50':>7}  {'CP-AUC':>7}  {'time':>6}")
    print("─" * 66)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, diverged = _train_step(
            model, optimizer, ctx, ctx.train_data,
            args.neg_k, rng, args.neg_k_cp, exclude_pairs,
            grad_clip=args.grad_clip,
        )
        if diverged:
            print(f"Epoch {epoch}: loss diverged — stopping.")
            break

        val_cp_auc, val_cp_ap, val_loss, test_loss, ph10, ph50 = _evaluate(
            model.gnn, model.predictor, ctx, args, epoch, exclude_pairs,
        )
        dt = time.time() - t0

        history["train_loss"].append(loss)
        history["val_ph10"].append(ph10)
        history["val_ph50"].append(ph50)
        history["val_cp_auc"].append(val_cp_auc)
        history["val_cp_ap"].append(val_cp_ap)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)

        if epoch % args.log_every == 0:
            print(f"{epoch:>6d}  {loss:>7.4f}  {val_loss:>7.4f}  {test_loss:>8.4f} "
                  f"{ph10:>7.4f}  {ph50:>7.4f}  {val_cp_auc:>7.4f}  {dt:>5.1f}s")

        if latest_path is not None and latest_path.exists():
            latest_path.unlink()
        latest_path = run_dir / f"{args.gnn_name}_L{args.num_layers}_latest_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch, "loss": loss, "val_ph50": ph50,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
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

    print(f"Checkpoint → {best_path}")
    print(f"Log        → {run_dir / 'train.log'}")
    print(f"History    → {history_path}")
    return model, ctx





if __name__ == "__main__":
    from utils import sample_training_negatives, collect_all_pairs
    import numpy as np
    from dataset import load_data
    from train import get_args, _train_step, _evaluate
    from models import build_model

    args = get_args()
    args.gnn_name = "gat"
    args.log_every = 20
    args.lr = 0.001
    args.num_layers = 4
    args.epochs = 200

    config_tag = f"{args.gnn_name}_L{args.num_layers}_h{args.hidden_dim}_{args.decoder}_lr{args.lr}"
    run_dir = RUNS_DIR / args.run_name / config_tag / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model, ctx = run(args, run_dir)

    exclude_pairs = collect_all_pairs(ctx)
    rng = np.random.default_rng(0)
    data = ctx.train_data
    lbl = data[ctx.target_edge].edge_label
    pos_ei = data[ctx.target_edge].edge_label_index[:, lbl == 1]
    eli, labels = sample_training_negatives(
        pos_ei, ctx.conv_idxs, neg_k_random=5, rng=rng, device=ctx.device,
        n_prot=data[ctx.src_type].num_nodes, neg_k_cp=5, exclude_pairs=exclude_pairs,
    )

    check_gradient_vanishing(model, ctx, data, eli, labels)
    check_input_gradient_flow(model, ctx, data, eli, labels)
    activations = check_oversmoothing(model, data)
    plot_node_metrics(activations, data.edge_index_dict, save_path = PLOTS_DIR / "metrics" / f"{args.gnn_name}_{args.num_layers}_metrics.png")
    plot_gradient_vanishing(model, ctx, data, eli, labels,
                        save_path= PLOTS_DIR / "gradnorm" / f"{args.gnn_name}_L{args.num_layers}_gradnorm.png")
    plot_input_gradient_flow(model, ctx, data, eli, labels,
                             save_path=PLOTS_DIR / "inputgrad" / f"{args.gnn_name}_L{args.num_layers}_inputgrad.png")


    plot_dirichlet_energy(activations, data.edge_index_dict, target_edge=ctx.target_edge, 
                        save_path= PLOTS_DIR / "dirichlet" / f"{args.gnn_name}_{args.num_layers}_dirichlet.png")
    plot_dirichlet_energy_grouped(activations, data.edge_index_dict, target_edge=ctx.target_edge,
                        save_path= PLOTS_DIR / "dirichlet" / f"{args.gnn_name}_{args.num_layers}_grouped_dirichlet.png")


