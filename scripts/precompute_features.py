"""Precompute mean-pooled features from activation files (parallel).

Activation files are ~16 MB compressed NPZs (np.savez_compressed),
and single-threaded decompression is the bottleneck (~5 sec/file).
With 120 cores available we parallelize via joblib.

Output: data/features/sycophancy/L{layer}_K{k}.npz
        keys: 'X' (n, 5120), 'qids' (n,)

Usage:
    python scripts/precompute_features.py [--n-jobs 32]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
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
    load_activations,
    mean_pool_at_percent,
    read_jsonl,
    resolve_layer_indices,
)


def process_one(
    qid: str, cache_dir: Path, layer_indices: list[int], k_percents: list[float]
) -> tuple[str, dict[tuple[int, float], np.ndarray] | None]:
    acts = load_activations(cache_dir / f"{qid}.npz", layer_indices)
    if acts is None:
        return qid, None
    out = {}
    for layer in layer_indices:
        arr = acts[layer].astype(np.float32)
        for k in k_percents:
            out[(layer, k)] = mean_pool_at_percent(arr, k)
    return qid, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=32)
    args = parser.parse_args()

    layer_indices = resolve_layer_indices(NUM_LAYERS, DEFAULT_LAYER_FRACS)
    k_percents = [float(k) for k in TRUNCATION_POINTS]

    print(f"Layers: {layer_indices}", flush=True)
    print(f"K%: {k_percents}", flush=True)
    print(f"Workers: {args.n_jobs}", flush=True)

    labeled_path = GENERATED_DIR / "sycophancy" / "labeled.jsonl"
    records = list(read_jsonl(labeled_path))
    qids = [r["question_id"] for r in records]
    print(f"Records: {len(qids)}", flush=True)

    cache_dir = ACTIVATIONS_DIR / "sycophancy"
    out_dir = DATA_DIR / "features" / "sycophancy"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=0)(
        delayed(process_one)(qid, cache_dir, layer_indices, k_percents)
        for qid in tqdm(qids, desc="precompute", unit="ex", mininterval=5.0)
    )

    # Assemble into matrices
    n = len(qids)
    matrices: dict[tuple, np.ndarray] = {}
    for layer in layer_indices:
        for k in k_percents:
            matrices[(layer, k)] = np.zeros((n, 5120), dtype=np.float32)

    valid_mask = np.zeros(n, dtype=bool)
    missing = 0
    for i, (qid, feats) in enumerate(results):
        if feats is None:
            missing += 1
            continue
        valid_mask[i] = True
        for key, vec in feats.items():
            matrices[key][i] = vec

    print(f"\nLoaded: {valid_mask.sum()}/{n}, missing: {missing}", flush=True)

    valid_qids = np.array([qid for i, qid in enumerate(qids) if valid_mask[i]])
    for (layer, k), mat in matrices.items():
        valid_mat = mat[valid_mask]
        out_path = out_dir / f"L{layer}_K{int(k)}.npz"
        np.savez(out_path, X=valid_mat, qids=valid_qids)
        print(f"  Saved {out_path.name}: {valid_mat.shape} ({valid_mat.nbytes / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
