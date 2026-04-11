"""Precompute mean-pooled features from activation files.

Bottleneck discovered: /dev/vda is a rotational HDD, so 48 parallel random
reads cause seek thrashing. Strategy:
  - Sort files by inode so reads are near-sequential on disk.
  - One thread reads raw bytes serially.
  - Thread pool decompresses in parallel (zlib/np.load release the GIL).

Output: data/features/sycophancy/L{layer}_K{k}.npz
        keys: 'X' (n, 5120), 'qids' (n,)

Usage:
    python scripts/precompute_features.py [--n-workers 16]
"""

import argparse
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ACTIVATIONS_DIR,
    DATA_DIR,
    DEFAULT_LAYER_FRACS,
    GENERATED_DIR,
    NUM_LAYERS,
    TRUNCATION_POINTS,
)
from src.lib import (
    mean_pool_at_percent,
    read_jsonl,
    resolve_layer_indices,
)


def decompress_and_pool(
    raw: bytes, layer_indices: list[int], k_percents: list[float]
) -> dict[tuple[int, float], np.ndarray] | None:
    try:
        npz = np.load(io.BytesIO(raw))
        files = set(npz.files)
        needed = {f"layer_{idx}" for idx in layer_indices}
        if not needed.issubset(files):
            return None
        out: dict[tuple[int, float], np.ndarray] = {}
        for layer in layer_indices:
            arr = npz[f"layer_{layer}"].astype(np.float32)
            for k in k_percents:
                out[(layer, k)] = mean_pool_at_percent(arr, k)
        return out
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=16)
    args = parser.parse_args()

    layer_indices = resolve_layer_indices(NUM_LAYERS, DEFAULT_LAYER_FRACS)
    k_percents = [float(k) for k in TRUNCATION_POINTS]

    print(f"Layers: {layer_indices}", flush=True)
    print(f"K%: {k_percents}", flush=True)
    print(f"Decomp workers: {args.n_workers}", flush=True)

    labeled_path = GENERATED_DIR / "sycophancy" / "labeled.jsonl"
    records = list(read_jsonl(labeled_path))
    qids = [r["question_id"] for r in records]
    n = len(qids)
    print(f"Records: {n}", flush=True)

    cache_dir = ACTIVATIONS_DIR / "sycophancy"
    out_dir = DATA_DIR / "features" / "sycophancy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-allocate output matrices
    matrices: dict[tuple, np.ndarray] = {}
    for layer in layer_indices:
        for k in k_percents:
            matrices[(layer, k)] = np.zeros((n, 5120), dtype=np.float32)
    valid_mask = np.zeros(n, dtype=bool)

    # Sort indices by inode to minimize HDD seek
    paths = [cache_dir / f"{q}.npz" for q in qids]
    existing = [(i, p) for i, p in enumerate(paths) if p.exists()]
    missing = n - len(existing)
    print(f"Missing files: {missing}", flush=True)

    existing.sort(key=lambda ip: os.stat(ip[1]).st_ino)

    # One reader thread streams bytes; pool decompresses.
    with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
        pending: list[tuple[int, object]] = []
        pbar = tqdm(total=len(existing), desc="precompute", unit="ex", mininterval=2.0)

        MAX_INFLIGHT = args.n_workers * 4

        def drain_one():
            nonlocal pending
            i, fut = pending.pop(0)
            feats = fut.result()
            if feats is not None:
                valid_mask[i] = True
                for key, vec in feats.items():
                    matrices[key][i] = vec
            pbar.update(1)

        for i, p in existing:
            with open(p, "rb") as f:
                raw = f.read()
            fut = pool.submit(decompress_and_pool, raw, layer_indices, k_percents)
            pending.append((i, fut))
            while len(pending) >= MAX_INFLIGHT:
                drain_one()

        while pending:
            drain_one()
        pbar.close()

    print(f"\nLoaded: {valid_mask.sum()}/{n}, missing: {n - valid_mask.sum()}", flush=True)

    valid_qids = np.array([qid for i, qid in enumerate(qids) if valid_mask[i]])
    for (layer, k), mat in matrices.items():
        valid_mat = mat[valid_mask]
        out_path = out_dir / f"L{layer}_K{int(k)}.npz"
        np.savez(out_path, X=valid_mat, qids=valid_qids)
        print(f"  Saved {out_path.name}: {valid_mat.shape} ({valid_mat.nbytes / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
