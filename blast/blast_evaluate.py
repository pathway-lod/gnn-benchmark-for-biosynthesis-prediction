#!/usr/bin/env python3
"""
Compute P-H@K for the BLASTp baseline on PlantMetBench (taxa holdout).

For each evaluation reaction c:
  score(p, c) = max BLASTp bitscore between p and any training catalyst of c
              = 0 if no training catalyst of c has a BLAST hit from p

Proteins are ranked by this score; ties broken by random order (seed 42).
P-H@K is computed exactly as in the GNN evaluation (eval_embedded_only=True,
ranking pool = 8,445 non-zero ESM-C proteins).

Method follows ReactZyme (Hua et al., 2024): reaction-conditioned sequence
retrieval, adapted for the reverse task direction (given reaction, rank proteins).

BLAST hits (blast/blast_results.tsv) are precomputed and shipped with this
repo, so this script only needs the standard PlantMetBench data release —
no local BLAST+ installation is required to reproduce the reported numbers.
Regenerating blast_results.tsv from raw sequences requires a local `blastp`/
`makeblastdb` install and is outside the scope of this script.

Run from repo root:
    python blast/blast_evaluate.py
    python blast/blast_evaluate.py --split val     # validation split
    python blast/blast_evaluate.py --dedup-pool    # 2,232 deduplicated pool
"""
import argparse
import json
import re
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BLAST_DIR = ROOT / "blast"

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["val", "test"], default="test")
ap.add_argument("--k-list", nargs="+", type=int, default=[1, 5, 10, 50])
ap.add_argument("--blast-results", default=str(BLAST_DIR / "blast_results.tsv"))
ap.add_argument("--dedup-pool", action="store_true",
                help="Evaluate on the 2,232 deduplicated ESM-C pool instead of 8,445.")
args = ap.parse_args()

# ── load graph + splits ──────────────────────────────────────────────────────
print("Loading heterodata and splits …")
hd     = torch.load(DATA / "heterodata.pt", weights_only=False)
splits = torch.load(DATA / "splits_taxa.pt", weights_only=False)

TARGET = ("Protein", "catalyzes", "Interaction")

train_ei = splits["train_edge_index"]   # [2, n_train]
eval_ei  = splits[f"{args.split}_edge_index"]  # [2, n_eval]

# ── protein ranking pool ─────────────────────────────────────────────────────
# Pool = proteins in embeddings_protein.pt (8,445 IRIs) — same as GNN evaluation
nodes_all = pd.read_csv(DATA / "nodes.tsv", sep="\t", low_memory=False)
prot_nodes_all = nodes_all[nodes_all["node_type"] == "Protein"].reset_index(drop=True)
iri_to_pidx = {row["node_id"]: i for i, row in prot_nodes_all.iterrows()}

prot_emb_dict = torch.load(DATA / "embeddings_protein.pt", weights_only=False)
pool_iris = [iri for iri in prot_emb_dict if iri in iri_to_pidx]
full_pool_set = set(iri_to_pidx[iri] for iri in pool_iris)

# Optionally collapse to 2,232 distinct ESM-C vectors.
# For each group of identical sequences, one representative is chosen and
# the group is scored by the max bitscore of any group member.
if args.dedup_pool:
    emb_list   = [prot_emb_dict[iri] for iri in pool_iris]
    emb_matrix = torch.stack(emb_list)
    # unique returns one row per distinct vector; inverse maps each row → group id
    _, inverse = torch.unique(emb_matrix, dim=0, return_inverse=True)
    # pick the first protein index seen per group as the representative
    group_to_rep: dict[int, int] = {}
    pidx_to_group: dict[int, int] = {}
    for iri, g in zip(pool_iris, inverse.tolist()):
        pidx = iri_to_pidx[iri]
        pidx_to_group[pidx] = g
        if g not in group_to_rep:
            group_to_rep[g] = pidx
    pool_set = set(group_to_rep.values())
    print(f"Ranking pool: {len(pool_set)} proteins (deduplicated ESM-C pool)")
else:
    pool_set      = full_pool_set
    pidx_to_group = None  # unused in full-pool mode
    group_to_rep  = None
    print(f"Ranking pool: {len(pool_set)} proteins (full ESM-C pool)")

n_pool = len(pool_set)

# pool index → rank position (for sorting)
pool_list = sorted(pool_set)
pool_pos  = {idx: pos for pos, idx in enumerate(pool_list)}

# ── training catalyst lookup: reaction_idx → set of training protein_idx ─────
train_p_idx = train_ei[0].tolist()
train_i_idx = train_ei[1].tolist()
reaction_to_train_catalysts: dict[int, set[int]] = defaultdict(set)
for p, i in zip(train_p_idx, train_i_idx):
    reaction_to_train_catalysts[i].add(p)

def safe_id(iri: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", iri)

# map safe_id → protein node index (reuse prot_nodes_all loaded above)
safe_to_idx: dict[str, int] = {safe_id(row["node_id"]): i
                                 for i, row in prot_nodes_all.iterrows()}

# ── parse BLAST results ───────────────────────────────────────────────────────
print(f"Parsing {args.blast_results} …")
# columns: qseqid  sseqid  bitscore
# qseqid format: safe_id(iri) + " idx=N"   (note: FASTA header space is cut at first space by BLAST)
# sseqid format: same

blast_scores: dict[tuple[int,int], float] = {}   # (query_idx, subject_idx) → bitscore
skipped = 0
with open(args.blast_results) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        qid, sid, bitscore = parts[0], parts[1], float(parts[2])

        # extract idx from the header token: "safe_iri_idx=N"
        # Note: BLAST cuts header at first space, so we embedded idx in the id itself
        # format from export: ">safe_id idx=N" → BLAST stores only "safe_id"
        # we need to re-derive idx from safe_id
        q_idx = safe_to_idx.get(qid)
        s_idx = safe_to_idx.get(sid)
        if q_idx is None or s_idx is None:
            skipped += 1
            continue
        key = (q_idx, s_idx)
        if key not in blast_scores or blast_scores[key] < bitscore:
            blast_scores[key] = bitscore

print(f"  Parsed {len(blast_scores):,} hits  ({skipped} skipped / unresolved)")

# Build subject-indexed lookup: subject_idx → {query_idx: bitscore}
# In dedup mode, subject is a group representative and we aggregate over the group.
# We first collect raw hits, then collapse by group when dedup_pool is active.
subj_to_query_raw: dict[int, dict[int, float]] = defaultdict(dict)
for (q_idx, s_idx), score in blast_scores.items():
    if q_idx in full_pool_set:
        subj_to_query_raw[s_idx][q_idx] = score  # blast_scores already holds max per pair

if args.dedup_pool:
    # Collapse: for each subject, aggregate query scores across all group members.
    # The score for representative r of a group G is max over g in G of bitscore(s, g).
    subj_to_query: dict[int, dict[int, float]] = defaultdict(dict)
    for s_idx, q_scores in subj_to_query_raw.items():
        for q_idx, bitscore in q_scores.items():
            rep = group_to_rep[pidx_to_group[q_idx]]
            if rep not in subj_to_query[s_idx] or subj_to_query[s_idx][rep] < bitscore:
                subj_to_query[s_idx][rep] = bitscore
else:
    subj_to_query = subj_to_query_raw

# ── evaluation ───────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
hits = {k: 0 for k in args.k_list}
n_queries = 0

eval_p_idx = eval_ei[0].tolist()
eval_i_idx = eval_ei[1].tolist()

# only score reactions where true catalyst is in the (full) pool
# In dedup mode, look up the group representative for the true protein.
def get_pool_target(p: int):
    """Return the pool index we expect to find in the top-K for protein p."""
    if args.dedup_pool:
        return group_to_rep.get(pidx_to_group.get(p)) if p in full_pool_set else None
    return p if p in pool_set else None

queries = [(p, i) for p, i in zip(eval_p_idx, eval_i_idx)
           if get_pool_target(p) is not None]
print(f"Evaluation queries ({args.split}): {len(queries)}")

for true_p, rxn_i in queries:
    target_p = get_pool_target(true_p)  # pool-representative to rank
    train_cats = reaction_to_train_catalysts.get(rxn_i, set())
    if not train_cats:
        scores = np.zeros(n_pool, dtype=np.float32)
    else:
        scores = np.zeros(n_pool, dtype=np.float32)
        for tc in train_cats:
            for q_idx, bitscore in subj_to_query.get(tc, {}).items():
                pos = pool_pos[q_idx]
                if bitscore > scores[pos]:
                    scores[pos] = bitscore

    # rank: descending bitscore; tie-break with random noise
    noise = rng.random(n_pool) * 1e-6
    order = np.argsort(-(scores + noise))
    ranked_pool = [pool_list[i] for i in order]
    rank = ranked_pool.index(target_p) + 1  # 1-indexed

    for k in args.k_list:
        if rank <= k:
            hits[k] += 1
    n_queries += 1

pool_label = f"{n_pool} proteins ({'dedup' if args.dedup_pool else 'full'} pool)"
print(f"\n── BLASTp baseline ({args.split} split, taxa holdout, {pool_label}) ──")
print(f"  Queries evaluated: {n_queries}")
for k in args.k_list:
    print(f"  P-H@{k:<3}: {hits[k]/n_queries*100:.2f}%  ({hits[k]}/{n_queries})")
