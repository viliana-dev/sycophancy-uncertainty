"""Experiment 4: Probe confidence trajectories by uncertainty group.

Question: How does the sycophancy signal accumulate through the CoT, and does
the shape differ between confident and uncertain records?

Method:
  - Train a single "reference" probe on L30 K100 features (train+val).
  - On the test split, apply the probe to L30 features at each K ∈ {10,25,50,75,100}.
    This gives a trajectory of P(sycophancy) vs depth-of-thinking per record.
  - Group records by (label × confidence regime), where confidence regime is
    token-entropy < / >= trainval median (same split as Exp 5).
  - Average trajectories within each of the 4 groups; report mean ± SE and
    save for plotting.

Hypothesis (from Exp 3/5):
  - Confident + syco: crisp rise — probe light up early and commits.
  - Uncertain + syco: flat then late jump — drift toward persona-favored option.
  - Non-syco: flat low.

Usage:
    python scripts/exp4_trajectories.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, GENERATED_DIR, RESULTS_DIR
from src.lib import read_jsonl


BEST_LAYER = 30
REF_K = 100
K_POINTS_THINKING = [10, 25, 50, 75, 100]


def load_features(layer: int, k_pct: int) -> tuple[np.ndarray, np.ndarray]:
    feat_dir = DATA_DIR / "features" / "sycophancy"
    npz = np.load(feat_dir / f"L{layer}_K{k_pct}.npz")
    X = npz["X"]
    qids = npz["qids"]
    npz.close()
    return X, qids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=BEST_LAYER)
    args = parser.parse_args()

    # ─── Load features for all K at the chosen layer ─────────────────────
    feats: dict[int, np.ndarray] = {}
    qids_ref = None

    # K=0 (pre-thinking) if available
    k0_path = DATA_DIR / "features" / "sycophancy" / f"L{args.layer}_K0.npz"
    has_k0 = k0_path.exists()
    if has_k0:
        X0, qids0 = load_features(args.layer, 0)
        qids_ref = qids0
        feats[0] = X0
        print(f"Loaded K=0 (pre-thinking) features: {X0.shape}", flush=True)

    for k in K_POINTS_THINKING:
        X, qids = load_features(args.layer, k)
        if qids_ref is None:
            qids_ref = qids
        else:
            assert np.array_equal(qids, qids_ref), f"qid order mismatch at K{k}"
        feats[k] = X

    K_POINTS = ([0] if has_k0 else []) + K_POINTS_THINKING
    qid_to_idx = {q: i for i, q in enumerate(qids_ref)}
    print(f"Loaded L{args.layer} features for K={K_POINTS}, N={len(qids_ref)}", flush=True)

    # ─── Labels, splits, entropy ──────────────────────────────────────────
    records = list(read_jsonl(GENERATED_DIR / "sycophancy" / "labeled.jsonl"))
    rec_lookup = {r["question_id"]: r for r in records}
    with open(GENERATED_DIR / "sycophancy" / "splits.json") as f:
        splits = json.load(f)

    ent_lookup = {}
    for rec in read_jsonl(GENERATED_DIR / "sycophancy" / "answer_entropy.jsonl"):
        ent_lookup[rec["question_id"]] = rec["entropy"]
    print(f"Loaded {len(ent_lookup)} entropy records", flush=True)

    # ─── Train reference probe on L30 K100 (trainval) ────────────────────
    trainval_qids = [q for q in splits["train"] + splits["val"] if q in qid_to_idx]
    test_qids = [q for q in splits["test"] if q in qid_to_idx]
    tv_idx = np.array([qid_to_idx[q] for q in trainval_qids])
    te_idx = np.array([qid_to_idx[q] for q in test_qids])

    X_tv = feats[REF_K][tv_idx]
    y_tv = np.array([rec_lookup[q]["label"] for q in trainval_qids])
    y_te = np.array([rec_lookup[q]["label"] for q in test_qids])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])
    pipe.fit(X_tv, y_tv)
    print(f"Reference probe: L{args.layer} K{REF_K}, trainval N={len(tv_idx)}", flush=True)

    # ─── Confidence split by token entropy (trainval median) ─────────────
    tv_ent = np.array([ent_lookup[q] for q in trainval_qids if q in ent_lookup])
    ent_median = float(np.median(tv_ent))
    print(f"Token-entropy median (trainval): {ent_median:.6f}", flush=True)

    # ─── Compute P(syco) on test at each K ───────────────────────────────
    probs_by_k: dict[int, np.ndarray] = {}
    for k in K_POINTS:
        X_te = feats[k][te_idx]
        probs_by_k[k] = pipe.predict_proba(X_te)[:, 1]

    # ─── Assign each test record to a group ──────────────────────────────
    # 0: conf_nonsyco, 1: conf_syco, 2: unc_nonsyco, 3: unc_syco
    group = np.full(len(test_qids), -1, dtype=int)
    for i, q in enumerate(test_qids):
        if q not in ent_lookup:
            continue
        confident = ent_lookup[q] <= ent_median
        syco = bool(rec_lookup[q]["label"])
        if confident and not syco:
            group[i] = 0
        elif confident and syco:
            group[i] = 1
        elif not confident and not syco:
            group[i] = 2
        elif not confident and syco:
            group[i] = 3

    group_names = {
        0: "confident_nonsyco",
        1: "confident_syco",
        2: "uncertain_nonsyco",
        3: "uncertain_syco",
    }

    trajectories: dict[str, dict] = {}
    print(f"\n{'Group':<22} {'N':<6} " + "  ".join([f"K{k:<4}" for k in K_POINTS]), flush=True)
    print("-" * 72, flush=True)
    for g_idx, name in group_names.items():
        mask = group == g_idx
        n = int(mask.sum())
        means = []
        ses = []
        for k in K_POINTS:
            vals = probs_by_k[k][mask]
            m = float(vals.mean()) if n else float("nan")
            se = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            means.append(round(m, 4))
            ses.append(round(se, 4))
        trajectories[name] = {"n": n, "K": K_POINTS, "mean": means, "se": ses}
        row = f"{name:<22} {n:<6} " + "  ".join([f"{m:.3f}" for m in means])
        print(row, flush=True)

    # Per-K gap (syco - nonsyco) within each regime, and cross-regime gap
    print(f"\nSyco − Non-syco gap (per regime, per K):", flush=True)
    for regime, syco_name, non_name in [
        ("confident", "confident_syco", "confident_nonsyco"),
        ("uncertain", "uncertain_syco", "uncertain_nonsyco"),
    ]:
        s = np.array(trajectories[syco_name]["mean"])
        n = np.array(trajectories[non_name]["mean"])
        gap = s - n
        print(f"  {regime:<10} " + "  ".join([f"K{k}:{g:+.3f}" for k, g in zip(K_POINTS, gap)]), flush=True)

    out = {
        "layer": args.layer,
        "reference_k": REF_K,
        "k_points": K_POINTS,
        "entropy_median": ent_median,
        "n_test_total": len(test_qids),
        "trajectories": trajectories,
    }
    out_path = RESULTS_DIR / "exp4_trajectories.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
