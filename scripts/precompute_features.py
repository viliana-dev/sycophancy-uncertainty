"""Precompute mean-pooled features from activation files.

Activation files are huge (~76 MB each, 5096 files = ~400 GB uncompressed)
because they store full thinking sequences. For linear probing we only need
mean-pooled vectors per (layer, K%) — much smaller (~2 GB total).

This script iterates once through all activation files, computes
mean_pool_at_percent for each (layer, K%), and saves a single NPZ per (layer, K%)
as a (n_records, hidden_dim) float32 matrix.

Output: data/features/sycophancy/L{layer}_K{k}.npz
        keys: 'X' (n, 5120), 'qids' (n,)

Usage:
    python scripts/precompute_features.py
"""

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ACTIVATIONS_DIR,
    DEFAULT_LAYER_FRACS,
    DATA_DIR,
    GENERATED_DIR,
    NUM_LAYERS,
    TRUNCATION_POINTS,
)
from src.lib import (
    load_activations,
    mean_pool_at_percent,
    read_jsonl,
    resolve_layer_indices,
)


def main():
    layer_indices = resolve_layer_indices(NUM_LAYERS, DEFAULT_LAYER_FRACS)
    k_percents = [float(k) for k in TRUNCATION_POINTS]

    print(f"Layers: {layer_indices}")
    print(f"K%: {k_percents}")

    labeled_path = GENERATED_DIR / "sycophancy" / "labeled.jsonl"
    records = list(read_jsonl(labeled_path))
    qids = [r["question_id"] for r in records]
    print(f"Records: {len(qids)}")

    cache_dir = ACTIVATIONS_DIR / "sycophancy"
    out_dir = DATA_DIR / "features" / "sycophancy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-allocate output matrices: {(layer, k_pct): np.array(n, 5120)}
    n = len(qids)
    matrices: dict[tuple, np.ndarray] = {}
    for layer in layer_indices:
        for k in k_percents:
            matrices[(layer, k)] = np.zeros((n, 5120), dtype=np.float32)

    valid_mask = np.zeros(n, dtype=bool)
    missing = 0

    for i, qid in enumerate(tqdm(qids, desc="precompute", unit="ex")):
        acts = load_activations(cache_dir / f"{qid}.npz", layer_indices)
        if acts is None:
            missing += 1
            continue
        valid_mask[i] = True

        for layer in layer_indices:
            arr = acts[layer].astype(np.float32)
            for k in k_percents:
                matrices[(layer, k)][i] = mean_pool_at_percent(arr, k)

    print(f"\nLoaded: {valid_mask.sum()}/{n}, missing: {missing}", flush=True)

    # Save per (layer, k_pct), keeping only valid rows
    valid_qids = np.array([qid for i, qid in enumerate(qids) if valid_mask[i]])
    for (layer, k), mat in matrices.items():
        valid_mat = mat[valid_mask]
        out_path = out_dir / f"L{layer}_K{int(k)}.npz"
        np.savez(out_path, X=valid_mat, qids=valid_qids)
        print(f"  Saved {out_path.name}: {valid_mat.shape} ({valid_mat.nbytes / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
