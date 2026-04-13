"""Precompute mean-pooled features from gpt-oss activation files.

Same logic as precompute_features.py but with gpt-oss paths and hidden_size.

Output: data/features/sycophancy_gptoss/L{layer}_K{k}.npz
        keys: 'X' (n, 2880), 'qids' (n,)

Usage:
    python scripts/precompute_features_gptoss.py [--n-workers 16]
"""

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ACTIVATIONS_DIR,
    DATA_DIR,
    GENERATED_DIR,
    GPTOSS_HIDDEN_SIZE,
    GPTOSS_LAYER_FRACS,
    GPTOSS_NUM_LAYERS,
    TRUNCATION_POINTS,
)
from src.lib import mean_pool_at_percent, read_jsonl, resolve_layer_indices


_LAYER_INDICES: list[int] = []
_K_PERCENTS: list[float] = []


def _init_worker(layer_indices, k_percents):
    global _LAYER_INDICES, _K_PERCENTS
    _LAYER_INDICES = layer_indices
    _K_PERCENTS = k_percents


def worker(indexed_path) -> tuple[int, dict[tuple[int, float], np.ndarray] | None]:
    i, p = indexed_path
    try:
        npz = np.load(p)
        files = set(npz.files)
        needed = {f"layer_{idx}" for idx in _LAYER_INDICES}
        if not needed.issubset(files):
            npz.close()
            return (i, None)
        out: dict[tuple[int, float], np.ndarray] = {}
        for layer in _LAYER_INDICES:
            arr = npz[f"layer_{layer}"].astype(np.float32)
            for k in _K_PERCENTS:
                out[(layer, k)] = mean_pool_at_percent(arr, k)
        npz.close()
        return (i, out)
    except Exception:
        return (i, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=16)
    args = parser.parse_args()

    layer_indices = resolve_layer_indices(GPTOSS_NUM_LAYERS, GPTOSS_LAYER_FRACS)
    k_percents = [float(k) for k in TRUNCATION_POINTS]

    print(f"Layers: {layer_indices}", flush=True)
    print(f"K%: {k_percents}", flush=True)

    labeled_path = GENERATED_DIR / "sycophancy_gptoss" / "labeled.jsonl"
    records = list(read_jsonl(labeled_path))
    qids = [r["question_id"] for r in records]
    n = len(qids)
    print(f"Records: {n}", flush=True)

    cache_dir = ACTIVATIONS_DIR / "sycophancy_gptoss"
    out_dir = DATA_DIR / "features" / "sycophancy_gptoss"
    out_dir.mkdir(parents=True, exist_ok=True)

    matrices = {}
    for layer in layer_indices:
        for k in k_percents:
            matrices[(layer, k)] = np.zeros((n, GPTOSS_HIDDEN_SIZE), dtype=np.float32)
    valid_mask = np.zeros(n, dtype=bool)

    paths = [cache_dir / f"{q}.npz" for q in qids]
    indexed = [(i, p) for i, p in enumerate(paths) if p.exists()]
    missing = n - len(indexed)
    print(f"Missing files: {missing}", flush=True)

    indexed.sort(key=lambda ip: os.stat(ip[1]).st_ino)
    print(f"Launching {args.n_workers} workers", flush=True)

    with mp.Pool(
        processes=args.n_workers,
        initializer=_init_worker,
        initargs=(layer_indices, k_percents),
    ) as pool:
        pbar = tqdm(total=len(indexed), desc="precompute", unit="ex", mininterval=5.0)
        for i, feats in pool.imap_unordered(worker, indexed, chunksize=8):
            if feats is not None:
                valid_mask[i] = True
                for key, vec in feats.items():
                    matrices[key][i] = vec
            pbar.update(1)
        pbar.close()

    print(f"\nLoaded: {valid_mask.sum()}/{n}", flush=True)

    valid_qids = np.array([qid for i, qid in enumerate(qids) if valid_mask[i]])
    for (layer, k), mat in matrices.items():
        valid_mat = mat[valid_mask]
        out_path = out_dir / f"L{layer}_K{int(k)}.npz"
        np.savez(out_path, X=valid_mat, qids=valid_qids)
        print(f"  Saved {out_path.name}: {valid_mat.shape}", flush=True)


if __name__ == "__main__":
    main()
