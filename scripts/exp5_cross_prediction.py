"""Experiment 5: Cross-prediction across uncertainty regimes.

Question: Is uncertain sycophancy a different computational mechanism than
confident sycophancy, or just a noisier version of the same one?

Method:
  - Split records into confident vs uncertain at the median uncertainty.
  - Train probe on confident half only → test on uncertain half (and vice versa).
  - Compare against within-distribution baselines (confident→confident, uncertain→uncertain).
  - Two large drops in cross-AUROC = two distinct mechanisms.
  - Symmetric performance = one mechanism, just noisier on one half.

Runs for both uncertainty metrics (lexical heuristic and token entropy)
and reports a 2x4 table per metric.

Usage:
    python scripts/exp5_cross_prediction.py
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
from src.evaluate import bootstrap_auroc_se, compute_auroc
from src.lib import read_jsonl


BEST_LAYER = 30
BEST_K = 100


def load_features(layer: int, k_pct: int) -> tuple[np.ndarray, np.ndarray]:
    feat_dir = DATA_DIR / "features" / "sycophancy"
    npz = np.load(feat_dir / f"L{layer}_K{k_pct}.npz")
    X = npz["X"]
    qids = npz["qids"]
    npz.close()
    return X, qids


def fit_eval(X_tr, y_tr, X_te, y_te, C=1.0) -> dict:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
    ])
    pipe.fit(X_tr, y_tr)
    probs = pipe.predict_proba(X_te)[:, 1]
    auroc = compute_auroc(y_te, probs)
    se = bootstrap_auroc_se(y_te, probs, n_boot=500)
    preds = (probs >= 0.5).astype(int)
    acc = float((preds == y_te).mean())
    return {"auroc": auroc, "auroc_se": se, "accuracy": acc, "n": int(len(y_te)),
            "n_pos": int(y_te.sum())}


def run_one_metric(
    metric_name: str, uncertainty_lookup: dict[str, float],
    X: np.ndarray, qids: np.ndarray,
    rec_lookup: dict, splits: dict,
) -> dict:
    print(f"\n=== Metric: {metric_name} ===", flush=True)

    # Map qid -> idx
    qid_to_idx = {q: i for i, q in enumerate(qids)}

    # Filter records to those with both features and uncertainty
    valid_qids = [q for q in qids if q in rec_lookup and q in uncertainty_lookup]
    print(f"Valid records: {len(valid_qids)}/{len(qids)}", flush=True)

    # Compute median uncertainty over the trainval pool only (avoid test leakage)
    trainval_qids = [q for q in (splits["train"] + splits["val"]) if q in qid_to_idx and q in uncertainty_lookup]
    trainval_unc = np.array([uncertainty_lookup[q] for q in trainval_qids])
    median = float(np.median(trainval_unc))
    print(f"Median uncertainty (trainval pool): {median:.6f}", flush=True)

    def split_by_unc(qid_list, low_first=True):
        confident, uncertain = [], []
        for q in qid_list:
            if q not in qid_to_idx or q not in uncertainty_lookup:
                continue
            u = uncertainty_lookup[q]
            if u <= median:
                confident.append(q)
            else:
                uncertain.append(q)
        return confident, uncertain

    train_conf, train_unc = split_by_unc(splits["train"] + splits["val"])
    test_conf, test_unc = split_by_unc(splits["test"])

    print(f"Train confident: {len(train_conf)}  uncertain: {len(train_unc)}", flush=True)
    print(f"Test  confident: {len(test_conf)}  uncertain: {len(test_unc)}", flush=True)

    def to_arrays(qid_list):
        idx = [qid_to_idx[q] for q in qid_list]
        Xs = X[idx]
        ys = np.array([rec_lookup[q]["label"] for q in qid_list])
        return Xs, ys

    X_tr_conf, y_tr_conf = to_arrays(train_conf)
    X_tr_unc, y_tr_unc = to_arrays(train_unc)
    X_te_conf, y_te_conf = to_arrays(test_conf)
    X_te_unc, y_te_unc = to_arrays(test_unc)

    results = {}
    # Within-distribution baselines
    results["conf->conf"] = fit_eval(X_tr_conf, y_tr_conf, X_te_conf, y_te_conf)
    results["unc->unc"] = fit_eval(X_tr_unc, y_tr_unc, X_te_unc, y_te_unc)
    # Cross
    results["conf->unc"] = fit_eval(X_tr_conf, y_tr_conf, X_te_unc, y_te_unc)
    results["unc->conf"] = fit_eval(X_tr_unc, y_tr_unc, X_te_conf, y_te_conf)

    print(f"\n{'Train→Test':<14} {'AUROC':<16} {'Acc':<8} {'N':<6} {'Pos%':<8}", flush=True)
    print("-" * 56, flush=True)
    for label in ["conf->conf", "unc->unc", "conf->unc", "unc->conf"]:
        r = results[label]
        pos = r["n_pos"] / r["n"] if r["n"] else 0.0
        print(
            f"{label:<14} {r['auroc']:.4f}±{r['auroc_se']:.4f}  "
            f"{r['accuracy']:.4f}  {r['n']:<6} {pos:.1%}",
            flush=True,
        )

    drop_conf2unc = results["conf->conf"]["auroc"] - results["conf->unc"]["auroc"]
    drop_unc2conf = results["unc->unc"]["auroc"] - results["unc->conf"]["auroc"]
    print(f"\nGeneralization drops:", flush=True)
    print(f"  conf→unc: {drop_conf2unc:+.4f} (negative = transfers well)", flush=True)
    print(f"  unc→conf: {drop_unc2conf:+.4f}", flush=True)

    return {
        "median": median,
        "n_train_conf": len(train_conf),
        "n_train_unc": len(train_unc),
        "n_test_conf": len(test_conf),
        "n_test_unc": len(test_unc),
        "results": {
            k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
            for k, v in results.items()
        },
        "generalization_drops": {
            "conf_to_unc": round(drop_conf2unc, 4),
            "unc_to_conf": round(drop_unc2conf, 4),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=BEST_LAYER)
    parser.add_argument("--k", type=int, default=BEST_K)
    args = parser.parse_args()

    # Load features for best (layer, K%)
    X, qids = load_features(args.layer, args.k)
    print(f"Features L{args.layer}_K{args.k}: X={X.shape}", flush=True)

    # Load labeled records and splits
    records = list(read_jsonl(GENERATED_DIR / "sycophancy" / "labeled.jsonl"))
    rec_lookup = {r["question_id"]: r for r in records}
    with open(GENERATED_DIR / "sycophancy" / "splits.json") as f:
        splits = json.load(f)

    # Heuristic uncertainty (already in labeled.jsonl)
    heuristic_lookup = {r["question_id"]: r["uncertainty_score"] for r in records}

    # Token entropy
    entropy_lookup = {}
    ent_path = GENERATED_DIR / "sycophancy" / "answer_entropy.jsonl"
    if ent_path.exists():
        for rec in read_jsonl(ent_path):
            entropy_lookup[rec["question_id"]] = rec["entropy"]
        print(f"Loaded {len(entropy_lookup)} entropy records", flush=True)

    out = {
        "layer": args.layer,
        "k_pct": args.k,
        "by_metric": {},
    }

    out["by_metric"]["heuristic"] = run_one_metric(
        "heuristic", heuristic_lookup, X, qids, rec_lookup, splits,
    )
    if entropy_lookup:
        out["by_metric"]["entropy"] = run_one_metric(
            "entropy", entropy_lookup, X, qids, rec_lookup, splits,
        )

    out_path = RESULTS_DIR / "exp5_cross_prediction.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
