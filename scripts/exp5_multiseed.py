"""Experiment 5 multi-seed: Cross-prediction robustness to random split.

Re-runs cross-prediction with 10 different stratified train/test splits
to compute seed-level confidence intervals. CPU only, ~30s total.

Usage:
    python scripts/exp5_multiseed.py
    python scripts/exp5_multiseed.py --model-family gptoss
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, GENERATED_DIR, RESULTS_DIR
from src.evaluate import compute_auroc
from src.lib import read_jsonl

BEST_LAYER_QWEN = 30
BEST_LAYER_GPTOSS = 18
BEST_LAYER_QWEN_MOE = 36
N_SEEDS = 10
TEST_SIZE = 0.15


def load_features(feat_dir, layer: int, k_pct: int):
    npz = np.load(feat_dir / f"L{layer}_K{k_pct}.npz")
    X, qids = npz["X"], npz["qids"]
    npz.close()
    return X, qids


def fit_auroc(X_tr, y_tr, X_te, y_te):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])
    pipe.fit(X_tr, y_tr)
    probs = pipe.predict_proba(X_te)[:, 1]
    return compute_auroc(y_te, probs)


def run_multiseed(X, y, uncertainty, n_seeds=N_SEEDS):
    """Run cross-prediction with n_seeds different stratified splits."""
    all_results = {k: [] for k in ["conf->conf", "unc->unc", "conf->unc", "unc->conf"]}
    all_drops = {"conf_to_unc": [], "unc_to_conf": []}

    for seed in range(n_seeds):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))

        # Compute median on train only
        train_unc = uncertainty[train_idx]
        median = np.median(train_unc)

        # Split train by uncertainty
        tr_conf_mask = train_unc <= median
        tr_unc_mask = train_unc > median
        tr_conf_idx = train_idx[tr_conf_mask]
        tr_unc_idx = train_idx[tr_unc_mask]

        # Split test by uncertainty (using train median)
        test_unc_vals = uncertainty[test_idx]
        te_conf_mask = test_unc_vals <= median
        te_unc_mask = test_unc_vals > median
        te_conf_idx = test_idx[te_conf_mask]
        te_unc_idx = test_idx[te_unc_mask]

        # Skip if any group is too small
        if min(len(tr_conf_idx), len(tr_unc_idx), len(te_conf_idx), len(te_unc_idx)) < 20:
            continue

        # 4 cross-prediction configs
        cc = fit_auroc(X[tr_conf_idx], y[tr_conf_idx], X[te_conf_idx], y[te_conf_idx])
        uu = fit_auroc(X[tr_unc_idx], y[tr_unc_idx], X[te_unc_idx], y[te_unc_idx])
        cu = fit_auroc(X[tr_conf_idx], y[tr_conf_idx], X[te_unc_idx], y[te_unc_idx])
        uc = fit_auroc(X[tr_unc_idx], y[tr_unc_idx], X[te_conf_idx], y[te_conf_idx])

        all_results["conf->conf"].append(cc)
        all_results["unc->unc"].append(uu)
        all_results["conf->unc"].append(cu)
        all_results["unc->conf"].append(uc)
        all_drops["conf_to_unc"].append(cc - cu)
        all_drops["unc_to_conf"].append(uu - uc)

        print(f"  seed {seed}: cc={cc:.3f} uu={uu:.3f} cu={cu:.3f} uc={uc:.3f} "
              f"drop_cu={cc-cu:+.3f} drop_uc={uu-uc:+.3f}", flush=True)

    # Aggregate
    summary = {}
    for k, vals in all_results.items():
        arr = np.array(vals)
        summary[k] = {"mean": round(float(arr.mean()), 4),
                       "std": round(float(arr.std()), 4),
                       "values": [round(v, 4) for v in vals]}
    for k, vals in all_drops.items():
        arr = np.array(vals)
        summary[f"drop_{k}"] = {"mean": round(float(arr.mean()), 4),
                                 "std": round(float(arr.std()), 4),
                                 "values": [round(v, 4) for v in vals]}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-family", choices=["qwen", "gptoss", "qwen_moe"], default="qwen")
    args = parser.parse_args()

    is_gptoss = args.model_family == "gptoss"
    is_qwen_moe = args.model_family == "qwen_moe"
    suffix = "_gptoss" if is_gptoss else ("_qwen_moe" if is_qwen_moe else "")
    feat_dir = DATA_DIR / "features" / f"sycophancy{suffix}"
    gen_dir = GENERATED_DIR / f"sycophancy{suffix}"
    if is_gptoss:
        best_layer = BEST_LAYER_GPTOSS
    elif is_qwen_moe:
        best_layer = BEST_LAYER_QWEN_MOE
    else:
        best_layer = BEST_LAYER_QWEN

    # Load features
    X, qids = load_features(feat_dir, best_layer, 100)
    print(f"Features: X={X.shape}", flush=True)

    # Load labels and entropy
    records = list(read_jsonl(gen_dir / "labeled.jsonl"))
    rec_lookup = {r["question_id"]: r for r in records}

    entropy_lookup = {}
    ent_path = gen_dir / "answer_entropy.jsonl"
    if ent_path.exists():
        for rec in read_jsonl(ent_path):
            entropy_lookup[rec["question_id"]] = rec["entropy"]

    # Build aligned arrays
    qid_list = [q for q in qids if q in rec_lookup and q in entropy_lookup]
    qid_to_idx = {q: i for i, q in enumerate(qids)}
    idx = np.array([qid_to_idx[q] for q in qid_list])
    X_aligned = X[idx]
    y_aligned = np.array([rec_lookup[q]["label"] for q in qid_list])
    unc_aligned = np.array([entropy_lookup[q] for q in qid_list])

    print(f"Aligned: {len(qid_list)} records, {y_aligned.sum():.0f} positive", flush=True)

    print(f"\n=== Multi-seed cross-prediction ({N_SEEDS} seeds) ===", flush=True)
    summary = run_multiseed(X_aligned, y_aligned, unc_aligned, N_SEEDS)

    # Print summary
    print(f"\n{'Config':<16} {'Mean':>8} {'Std':>8}", flush=True)
    print("-" * 36, flush=True)
    for k in ["conf->conf", "unc->unc", "conf->unc", "unc->conf",
              "drop_conf_to_unc", "drop_unc_to_conf"]:
        s = summary[k]
        print(f"{k:<16} {s['mean']:>8.4f} {s['std']:>8.4f}", flush=True)

    # Save
    out_path = RESULTS_DIR / f"exp5_multiseed{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        model_name = "gpt-oss-20b" if is_gptoss else ("Qwen3-30B-A3B" if is_qwen_moe else "Qwen3-14B")
        json.dump({"model": model_name,
                    "n_seeds": N_SEEDS, "test_size": TEST_SIZE,
                    "layer": best_layer, "k_pct": 100,
                    "summary": summary}, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
