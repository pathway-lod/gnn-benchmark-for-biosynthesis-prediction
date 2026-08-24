#!/usr/bin/env python3
"""Extract EC class distribution from embeddings_ec.pt → results/ec_distribution.json

The EC embedding file stores a [n_conversions × 237] tensor:
  dims 0–7   : L1 one-hot (7 EC classes + unknown)
  dims 8–68  : L2 one-hot (60 sub-classes + unknown)
  dims 69–236: L3 one-hot (167 sub-sub-classes + unknown)

We read only the L1 slice to count how many Conversion nodes belong to each
top-level EC class, then write results/ec_distribution.json for the appendix table.

Usage:
    cd plantmetbench/
    python scripts/extract_ec_distribution.py
    python scripts/extract_ec_distribution.py --data-dir /path/to/data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

EC_CLASSES = [
    "Oxidoreductases",
    "Transferases",
    "Hydrolases",
    "Lyases",
    "Isomerases",
    "Ligases",
    "Translocases",
    "Unknown / unannotated",
]

# L1 slice boundaries
L1_START = 0
L1_END   = 8  # 7 classes + 1 unknown


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir",    default=str(Path(__file__).parent.parent / "data"))
    ap.add_argument("--results-dir", default=str(Path(__file__).parent.parent / "results"))
    args = ap.parse_args()

    ec_path = Path(args.data_dir) / "embeddings_ec.pt"
    if not ec_path.exists():
        print(f"ERROR: {ec_path} not found.")
        print("Run `python training/dataset.py` with --ec-features to download/generate it,")
        print("or download embeddings_ec.pt from Zenodo (DOI 10.5281/zenodo.21237830).")
        raise SystemExit(1)

    print(f"Loading {ec_path} ...")
    data = torch.load(ec_path, map_location="cpu", weights_only=False)

    # File format: {node_uri_str: 237-dim float32 tensor}
    if isinstance(data, dict):
        node_ids = list(data.keys())
        tensor   = torch.stack(list(data.values())).float()
    else:
        tensor   = data.float()
        node_ids = None

    n_conversions, n_dims = tensor.shape
    print(f"Tensor shape: {n_conversions} × {n_dims}")
    assert n_dims >= L1_END, f"Expected ≥{L1_END} dims, got {n_dims}"

    l1_slice = tensor[:, L1_START:L1_END]  # [n_conversions, 8]
    # Each row is one-hot → argmax gives the active class index
    class_indices = l1_slice.argmax(dim=1).tolist()

    counts = {name: 0 for name in EC_CLASSES}
    for idx in class_indices:
        counts[EC_CLASSES[idx]] += 1

    n_annotated = n_conversions - counts["Unknown / unannotated"]

    out = {
        "n_conversion_nodes": n_conversions,
        "n_annotated": n_annotated,
        "n_unannotated": counts["Unknown / unannotated"],
        "annotation_coverage_pct": round(100 * n_annotated / n_conversions, 1),
        "counts_by_ec_class": counts,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "ec_distribution.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nEC class distribution:")
    for cls, cnt in counts.items():
        pct = 100 * cnt / n_conversions
        print(f"  {cls:<30s}: {cnt:>5}  ({pct:4.1f}%)")
    print(f"\nAnnotated: {n_annotated} / {n_conversions} "
          f"({out['annotation_coverage_pct']}%)")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
