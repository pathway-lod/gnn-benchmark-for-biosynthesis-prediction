"""Drop alias Metabolite/GeneProduct nodes and blank-subtype Interaction nodes
from a GraphContext, with every index-referencing field correctly remapped.
"""
import torch
from dataset import GraphContext, load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_clean_ctx(ctx: GraphContext) -> GraphContext:
    """Return a new GraphContext with:
      - Metabolite / GeneProduct nodes whose node_id is a pathway-local alias
        (URI contains "/Pathway/") dropped -- confirmed 100% zero-feature.
      - Interaction nodes not tagged interaction_subtype == "Conversion" dropped
        (i.e. the blank-subtype duplicates) -- confirmed 100% zero-feature.

    HeteroData.subgraph() remaps the standard `edge_index` per edge type
    correctly, but does NOT remap `edge_label_index` (verified empirically --
    it leaves the raw old-numbering values in place while the node count
    shrinks, producing out-of-bounds indices). That field is remapped here
    manually, along with conv_idxs / conv_idxs_tensor / all_catalyst_lookup.
    """
    device = ctx.device

    def keep_mask_non_alias(node_type: str) -> torch.Tensor:
        sub = ctx.nodes_df[ctx.nodes_df.node_type == node_type].reset_index(drop=True)
        is_alias = sub["node_id"].str.contains("/Pathway/", regex=False)
        return torch.tensor((~is_alias).to_numpy(), dtype=torch.bool, device=device)

    inter = ctx.nodes_df[ctx.nodes_df.node_type == "Interaction"].reset_index(drop=True)
    keep_interaction = torch.tensor(
        (inter["interaction_subtype"].fillna("") == "Conversion").to_numpy(),
        dtype=torch.bool, device=device,
    )
    subset_dict = {
        "Metabolite":  keep_mask_non_alias("Metabolite"),
        "GeneProduct": keep_mask_non_alias("GeneProduct"),
        "Interaction": keep_interaction,
    }
    # Any other node type present (Protein always; Organism too if the caller's
    # load_data(remove_organism=False) kept it, or anything else future data
    # releases might add) must still get an explicit all-True mask on the
    # right device -- PyG's subgraph() falls back to torch.arange(num_nodes)
    # (no device= given, always CPU) for any node type *absent* from
    # subset_dict, which then mismatches our CUDA masks and edge_index.
    for nt in ctx.train_data.node_types:
        if nt not in subset_dict:
            subset_dict[nt] = torch.ones(ctx.train_data[nt].num_nodes, dtype=torch.bool, device=device)

    # old_idx -> new_idx lookup for Interaction (the only type edge_label_index
    # references that changes size; the Protein side of edge_label_index is
    # untouched since Protein is never filtered).
    n_inter_old = ctx.train_data["Interaction"].num_nodes
    old_to_new_inter = torch.full((n_inter_old,), -1, dtype=torch.long, device=device)
    old_to_new_inter[keep_interaction] = torch.arange(int(keep_interaction.sum()), device=device)

    TARGET_EDGE = ctx.target_edge  # ("Protein", "catalyzes", "Interaction")

    def clean_split(data):
        d_clean = data.clone().subgraph(subset_dict)

        # subgraph() empties an edge type's edge_index to shape (2, 0) but keeps
        # the key present in the graph.
        empty_et = [et for et in d_clean.edge_types if d_clean[et].edge_index.shape[1] == 0]
        for et in empty_et:
            del d_clean[et]

        eli = data[TARGET_EDGE].edge_label_index
        lbl = data[TARGET_EDGE].edge_label
        new_dst = old_to_new_inter[eli[1]]
        valid = new_dst >= 0  # should be all True: catalyzes always targets Conversion, which we keep
        if not valid.all():
            raise RuntimeError(
                f"{(~valid).sum().item()} label edges pointed to a dropped Interaction "
                "node -- unexpected, since catalyzes should only ever target Conversion."
            )
        d_clean[TARGET_EDGE].edge_label_index = torch.stack([eli[0][valid], new_dst[valid]])
        d_clean[TARGET_EDGE].edge_label = lbl[valid]
        return d_clean

    train_data = clean_split(ctx.train_data)
    val_data   = clean_split(ctx.val_data)
    test_data  = clean_split(ctx.test_data)

    # conv_idxs: trivial now -- every remaining Interaction node IS Conversion-subtype.
    n_conv_new = train_data["Interaction"].num_nodes
    conv_idxs = list(range(n_conv_new))
    conv_idxs_tensor = torch.arange(n_conv_new, dtype=torch.long, device=device)

    # all_catalyst_lookup: {old_conv_idx -> {protein_idxs}} -- remap keys to new indexing.
    all_catalyst_lookup = {
        int(old_to_new_inter[old_conv_idx]): prot_idxs
        for old_conv_idx, prot_idxs in ctx.all_catalyst_lookup.items()
        if old_to_new_inter[old_conv_idx] >= 0
    }

    # split_ei: {"train"/"val"/"test" -> [2, N] (protein_idx, interaction_idx) positive
    # pairs}. Protein side is untouched (never filtered); Interaction side needs the
    # same old_to_new_inter remap as edge_label_index above.
    split_ei = {}
    for name, ei in ctx.split_ei.items():
        ei = ei.to(device)  # ctx.split_ei comes straight from torch.load(), always on CPU
        new_dst = old_to_new_inter[ei[1]]
        valid = new_dst >= 0
        if not valid.all():
            raise RuntimeError(
                f"split_ei[{name!r}]: {(~valid).sum().item()} pairs pointed to a dropped "
                "Interaction node -- unexpected, since catalyzes should only ever target Conversion."
            )
        split_ei[name] = torch.stack([ei[0][valid], new_dst[valid]])

    return GraphContext(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        target_edge=ctx.target_edge,
        src_type=ctx.src_type,
        dst_type=ctx.dst_type,
        nodes_df=ctx.nodes_df,              # positional indices for Interaction/Metabolite/GeneProduct no longer match this df
        conv_idxs=conv_idxs,
        conv_idxs_tensor=conv_idxs_tensor,
        embedded_protein_mask=ctx.embedded_protein_mask,  # Protein untouched, still valid as-is
        all_catalyst_lookup=all_catalyst_lookup,
        device=ctx.device,
        data_dir=ctx.data_dir,
        split_ei=split_ei,
    )


def randomize_zero_features(clean_ctx: GraphContext, node_types=("GeneProduct", "Metabolite", "Protein"),
                             strategy: str = "random", seed: int = 42) -> GraphContext:
    """In-place: replace remaining all-zero x rows for the given node types
    (default: the three that still have zero-x nodes after build_clean_ctx()

    strategy:
      "random" -- torch.randn per missing node - each missing node gets an arbitrary, uninformative
        vector unrelated to the others).
      "mean"   -- every missing node gets the SAME vector: the mean of that
        type's observed (non-zero) embeddings in train_data. Places missing
        nodes at the modality's "center of mass" -- a neutral "typical node
        of this type" prior rather than a null vector, and better-behaved
        than zeros for attention/normalization-sensitive layers (GAT, HGT,
        any LayerNorm). Trade-off: it fabricates plausible-looking features,
        and because every missing node of a type collapses onto the exact
        same point, the model can (and likely will) learn "is this the mean
        vector" as a cheap, weakly-predictive proxy for "is this node's
        embedding actually missing" -- a shortcut signal unrelated to real
        biology, not a fix for the missingness itself.

    Does NOT touch Interaction by default: after build_clean_ctx(), every
    Interaction node's zero-ness reflects a genuinely missing MAP4
    fingerprint for a real Conversion reaction (not a duplicate we already
    dropped)
    """
    if strategy not in ("random", "mean"):
        raise ValueError(f"strategy must be 'random' or 'mean', got {strategy!r}")

    device = clean_ctx.device
    g = torch.Generator(device=device).manual_seed(seed)

    for nt in node_types:
        x_ref = clean_ctx.train_data[nt].x
        is_zero = (x_ref.abs().sum(dim=1) == 0)
        n_zero = int(is_zero.sum())
        if n_zero == 0:
            continue
        dim = x_ref.shape[1]

        if strategy == "random":
            fill = torch.randn(n_zero, dim, device=device, generator=g)
        else:  # "mean"
            observed_mean = x_ref[~is_zero].mean(dim=0)  # single [dim] vector
            fill = observed_mean.unsqueeze(0).expand(n_zero, -1)

        for data in (clean_ctx.train_data, clean_ctx.val_data, clean_ctx.test_data):
            data[nt].x[is_zero] = fill
        print(f"  {nt:<12}: replaced {n_zero:,}/{x_ref.shape[0]:,} zero-x rows ({strategy})")

    return clean_ctx


def load_ctx_for_run(hp: dict, data_dir, **overrides) -> GraphContext:
    """Rebuild the graph a train.py run used, from its saved hparams (vars(args)).

    Runs from before the data-cleaning merge have no data_type /
    remove_metabolite_organism_edges entries and rebuild as the raw graph.
    """
    hp = {**hp, **overrides}
    ctx = load_data(
        data_dir=data_dir,
        random_seed=hp.get("seed", 42),
        remove_is_part_of=not hp.get("keep_pathways", False),
        embedded_only_ranking=True,
        disjoint_train_ratio=hp.get("disjoint_train_ratio", 0.2),
        keep_catalyzed_by=hp.get("keep_catalyzed_by", False),
        load_ec_embeddings=hp.get("ec_features", False),
        remove_currency_metabolites=hp.get("remove_currency_metabolites", False),
        remove_all_metabolites=hp.get("remove_all_metabolites", False),
        organism_embeddings_path=hp.get("organism_embeddings_path"),
        organism_embedding_type=hp.get("organism_embedding_type", "mds"),
        remove_gene_organism_edges=hp.get("remove_gene_organism_edges", False),
        remove_organism_nodes=hp.get("remove_organism_nodes", False),
        remove_metabolite_organism_edges=hp.get("remove_metabolite_organism_edges", False),
        split_type=hp.get("split_type", "taxa"),
        species_pool=hp.get("species_pool", False),
        download=hp.get("download", False),
        print_summary=hp.get("print_dataset_summary", False),
    )
    data_type = hp.get("data_type", "raw")
    if data_type != "raw":
        ctx = build_clean_ctx(ctx)
        if data_type in ("mean", "random"):
            ctx = randomize_zero_features(ctx, strategy=data_type)
    return ctx


def interaction_node_ids(nodes_df, data_type: str = "raw") -> list[str]:
    """Interaction node_ids in the order of the ctx's Interaction indices."""
    inter = nodes_df[nodes_df.node_type == "Interaction"]
    if data_type != "raw":
        inter = inter[inter["interaction_subtype"].fillna("") == "Conversion"]
    return inter["node_id"].tolist()


if __name__ == "__main__":
    from models import build_model

    ctx = load_data(print_summary=False)
    clean_ctx = build_clean_ctx(ctx)

    print("node counts (train_data):")
    for nt in clean_ctx.train_data.node_types:
        print(f"  {nt:<12}: {clean_ctx.train_data[nt].num_nodes:,}")

    TARGET_EDGE = clean_ctx.target_edge
    for split_name, data in [("train", clean_ctx.train_data), ("val", clean_ctx.val_data), ("test", clean_ctx.test_data)]:
        eli = data[TARGET_EDGE].edge_label_index
        n_nodes = data["Interaction"].num_nodes
        in_bounds = (eli[1] < n_nodes).all().item()
        print(f"{split_name}: edge_label_index shape={tuple(eli.shape)}  all in-bounds={in_bounds}")

    print(f"\nconv_idxs_tensor length: {len(clean_ctx.conv_idxs_tensor):,}")
    print(f"all_catalyst_lookup entries: {len(clean_ctx.all_catalyst_lookup):,}")

    print("\nImputing remaining zero-x rows (strategy='mean'):")
    clean_ctx = randomize_zero_features(clean_ctx, strategy="mean")
    for nt in clean_ctx.train_data.node_types:
        xt = clean_ctx.train_data[nt].x
        n_zero_after = (xt.abs().sum(dim=1) == 0).sum().item()
        print(f"  {nt:<12}: zero-x rows remaining = {n_zero_after}")

    # end-to-end smoke test: does the existing build_model()/forward pass work unmodified?
    model = build_model(clean_ctx, gnn_name="sage", hidden_dim=64, num_layers=2)
    data = clean_ctx.train_data
    eli = data[TARGET_EDGE].edge_label_index
    logits = model(clean_ctx, data.x_dict, data.edge_index_dict, eli)
    print(f"\nSmoke test forward pass OK -- logits shape: {tuple(logits.shape)}")
