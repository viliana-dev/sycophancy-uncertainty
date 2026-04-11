"""Experiment 3: Activation probes stratified by uncertainty.

Uses precomputed features (run scripts/precompute_features.py first).

Questions:
1. Can a linear probe detect sycophancy from activations? (overall AUROC)
2. Does probe performance differ across uncertainty levels?
   - Higher AUROC on uncertain → stronger signal when model "surrenders"
   - Lower AUROC on uncertain → uncertain sycophancy is more subtle
   - Same AUROC → one mechanism regardless of uncertainty

Usage:
    python scripts/precompute_features.py   # one-time: ~30 min
    python scripts/exp3_probe_by_uncertainty.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA_DIR,
    DEFAULT_LAYER_FRACS,
    GENERATED_DIR,
    NUM_LAYERS,
    RESULTS_DIR,
    TRUNCATION_POINTS,
)
from src.evaluate import bootstrap_auroc_se, compute_auroc, compute_gmean2
from src.lib import read_jsonl, resolve_layer_indices


def load_features(layer: int, k_pct: float) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed (X, qids) for a single (layer, K%) config."""
    feat_dir = DATA_DIR / "features" / "sycophancy"
    npz = np.load(feat_dir / f"L{layer}_K{int(k_pct)}.npz")
    X = npz["X"]
    qids = npz["qids"]
    npz.close()
    return X, qids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-quintiles", type=int, default=5)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--uncertainty-source", choices=["heuristic", "entropy"], default="heuristic",
        help="heuristic: lexical score from labeled.jsonl; entropy: answer_entropy.jsonl",
    )
    parser.add_argument(
        "--stratify", choices=["percentile", "rank"], default="percentile",
        help="rank uses ordinal ranks (good when scores have many ties, e.g. token entropy)",
    )
    args = parser.parse_args()

    layer_indices = resolve_layer_indices(NUM_LAYERS, DEFAULT_LAYER_FRACS)
    k_percents = [float(k) for k in TRUNCATION_POINTS]

    # Include K=0 (pre-thinking) if feature files exist
    k0_exists = all(
        (DATA_DIR / "features" / "sycophancy" / f"L{layer}_K0.npz").exists()
        for layer in layer_indices
    )
    if k0_exists:
        k_percents = [0.0] + k_percents
        print("K=0 (pre-thinking) features found — including in sweep", flush=True)

    # ─── Load labeled data + splits ──────────────────────────────────────
    labeled_path = GENERATED_DIR / "sycophancy" / "labeled.jsonl"
    splits_path = GENERATED_DIR / "sycophancy" / "splits.json"

    if not labeled_path.exists() or not splits_path.exists():
        print("Run prepare_labeled.py first")
        return

    records = list(read_jsonl(labeled_path))
    with open(splits_path) as f:
        splits = json.load(f)

    rec_lookup = {r["question_id"]: r for r in records}

    n_syco = sum(r["label"] for r in records)
    print(f"Loaded {len(records)} labeled records ({n_syco} syco, {100*n_syco/len(records):.1f}%)", flush=True)

    # Optionally override uncertainty_score with token entropy
    if args.uncertainty_source == "entropy":
        ent_path = GENERATED_DIR / "sycophancy" / "answer_entropy.jsonl"
        if not ent_path.exists():
            print(f"Missing {ent_path} — run compute_answer_entropy.py first")
            return
        ent_lookup = {}
        for rec in read_jsonl(ent_path):
            ent_lookup[rec["question_id"]] = rec["entropy"]
        n_overridden = 0
        for qid, r in rec_lookup.items():
            if qid in ent_lookup:
                r["uncertainty_score"] = ent_lookup[qid]
                n_overridden += 1
        print(f"Override uncertainty_score with answer entropy: {n_overridden}/{len(rec_lookup)}", flush=True)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # ─── Sweep all (layer, K%) configs ───────────────────────────────────
    print(f"\n=== Training linear probes ===", flush=True)
    print(f"{'Config':<14} {'TestAUROC':<12} {'Acc':<8} {'gmean2':<8}", flush=True)
    print("-" * 45, flush=True)

    probe_results = {}  # {(layer, k): {pipe, test_auroc, ...}}
    test_qid_lookup = {}  # cache of test qids per (layer, k)

    for layer in layer_indices:
        for k in k_percents:
            X, qids = load_features(layer, k)

            # Build splits using qid index
            qid_to_idx = {q: i for i, q in enumerate(qids)}

            train_idx = [qid_to_idx[q] for q in splits["train"] if q in qid_to_idx]
            val_idx = [qid_to_idx[q] for q in splits["val"] if q in qid_to_idx]
            test_idx = [qid_to_idx[q] for q in splits["test"] if q in qid_to_idx]
            trainval_idx = train_idx + val_idx

            y = np.array([rec_lookup[q]["label"] for q in qids])

            X_tv = X[trainval_idx]
            y_tv = y[trainval_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(C=args.C, max_iter=1000, solver="lbfgs")),
            ])
            pipe.fit(X_tv, y_tv)

            test_probs = pipe.predict_proba(X_test)[:, 1]
            test_preds = pipe.predict(X_test)
            test_auroc = compute_auroc(y_test, test_probs)
            test_acc = float((test_preds == y_test).mean())
            test_g2 = compute_gmean2(y_test, test_preds)

            probe_results[(layer, k)] = {
                "pipe": pipe,
                "test_auroc": round(test_auroc, 4),
                "test_accuracy": round(test_acc, 4),
                "test_gmean2": round(test_g2, 4),
                "test_qids": np.array([qids[i] for i in test_idx]),
                "test_probs": test_probs,
                "test_labels": y_test,
            }

            print(f"L{layer}_K{int(k)}    {test_auroc:.4f}      {test_acc:.4f}  {test_g2:.4f}", flush=True)

    # Best config
    best_key = max(probe_results, key=lambda k: probe_results[k]["test_auroc"])
    best = probe_results[best_key]
    print(f"\nBest: L{best_key[0]} K{int(best_key[1])}: AUROC={best['test_auroc']:.4f}", flush=True)

    # ─── Stratify best probe by uncertainty quintiles ────────────────────
    print(f"\n=== Best probe per uncertainty quintile ===", flush=True)

    test_qids = best["test_qids"]
    test_probs = best["test_probs"]
    test_labels = best["test_labels"]

    uncertainties = np.array([rec_lookup[q]["uncertainty_score"] for q in test_qids])
    if args.stratify == "rank":
        order = uncertainties.argsort()
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        quintile_labels = (ranks * args.n_quintiles // len(uncertainties)).clip(0, args.n_quintiles - 1)
        edges = np.array([uncertainties[order[int(i * len(order) / args.n_quintiles)]]
                          for i in range(args.n_quintiles)] + [uncertainties.max()])
    else:
        edges = np.percentile(uncertainties, np.linspace(0, 100, args.n_quintiles + 1))
        quintile_labels = np.digitize(uncertainties, edges[1:-1])

    print(f"{'Quintile':<10} {'Range':<18} {'N':<6} {'Pos%':<8} {'AUROC':<14} {'Acc':<8}", flush=True)
    print("-" * 65, flush=True)

    quintile_results = {}
    for q_idx in range(args.n_quintiles):
        mask = quintile_labels == q_idx
        n = mask.sum()
        if n < 10:
            continue
        labels_q = test_labels[mask]
        probs_q = test_probs[mask]

        if len(np.unique(labels_q)) < 2:
            print(f"Q{q_idx+1:<9} (single class)", flush=True)
            continue

        auroc = compute_auroc(labels_q, probs_q)
        auroc_se = bootstrap_auroc_se(labels_q, probs_q, n_boot=500)
        preds_q = (probs_q >= 0.5).astype(int)
        acc = float((preds_q == labels_q).mean())
        pos_rate = float(labels_q.mean())

        lo = uncertainties[mask].min()
        hi = uncertainties[mask].max()
        print(
            f"Q{q_idx+1:<9} {lo:.1f}-{hi:.1f}{'':<10} "
            f"{n:<6} {pos_rate:.1%}{'':<4} "
            f"{auroc:.4f}±{auroc_se:.4f}  {acc:.4f}",
            flush=True,
        )
        quintile_results[f"Q{q_idx+1}"] = {
            "n": int(n),
            "n_pos": int(labels_q.sum()),
            "pos_rate": round(pos_rate, 4),
            "auroc": round(auroc, 4),
            "auroc_se": round(auroc_se, 4),
            "accuracy": round(acc, 4),
            "uncertainty_range": [float(lo), float(hi)],
        }

    # ─── Detailed: AUROC per (layer, K) per quintile ─────────────────────
    print(f"\n=== AUROC per (layer, K%) per quintile ===", flush=True)
    header = f"{'Config':<14} {'Overall':<10}"
    for q_idx in range(args.n_quintiles):
        header += f"{'Q'+str(q_idx+1):<10}"
    print(header, flush=True)
    print("-" * (14 + 10 + 10 * args.n_quintiles), flush=True)

    detailed = {}
    for (layer, k), res in sorted(probe_results.items()):
        row = f"L{layer}_K{int(k):<10}"[:14] + f" {res['test_auroc']:<10.4f}"
        config_data = {"overall": res["test_auroc"]}

        # Use the same test set ordering (per probe)
        t_qids = res["test_qids"]
        t_probs = res["test_probs"]
        t_labels = res["test_labels"]
        t_unc = np.array([rec_lookup[q]["uncertainty_score"] for q in t_qids])
        if args.stratify == "rank":
            t_order = t_unc.argsort()
            t_ranks = np.empty_like(t_order)
            t_ranks[t_order] = np.arange(len(t_order))
            t_quintile = (t_ranks * args.n_quintiles // len(t_unc)).clip(0, args.n_quintiles - 1)
        else:
            t_quintile = np.digitize(t_unc, edges[1:-1])  # use same edges as best

        for q_idx in range(args.n_quintiles):
            mask = t_quintile == q_idx
            if mask.sum() < 10 or len(np.unique(t_labels[mask])) < 2:
                row += f"{'N/A':<10}"
                config_data[f"Q{q_idx+1}"] = None
            else:
                auroc = compute_auroc(t_labels[mask], t_probs[mask])
                row += f"{auroc:<10.4f}"
                config_data[f"Q{q_idx+1}"] = round(auroc, 4)

        print(row, flush=True)
        detailed[f"L{layer}_K{int(k)}"] = config_data

    # ─── Save results ─────────────────────────────────────────────────────
    out = {
        "n_records": len(records),
        "layer_indices": layer_indices,
        "k_percents": k_percents,
        "best_config": {
            "layer": best_key[0],
            "k_pct": best_key[1],
            "test_auroc": best["test_auroc"],
            "test_accuracy": best["test_accuracy"],
        },
        "probe_results": {
            f"L{k[0]}_K{int(k[1])}": {
                "test_auroc": v["test_auroc"],
                "test_accuracy": v["test_accuracy"],
                "test_gmean2": v["test_gmean2"],
            }
            for k, v in probe_results.items()
        },
        "quintile_results": quintile_results,
        "detailed_by_quintile": detailed,
        "uncertainty_edges": edges.tolist(),
    }

    out_name = "exp3_probe_by_uncertainty"
    if args.uncertainty_source != "heuristic":
        out_name += f"_{args.uncertainty_source}"
    if args.stratify != "percentile":
        out_name += f"_{args.stratify}"
    out_path = RESULTS_DIR / f"{out_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
