"""Experiment 3: Activation probes stratified by uncertainty.

Questions:
1. Can a linear probe detect sycophancy from activations? (overall AUROC)
2. Does probe performance differ across uncertainty levels?
   - Higher AUROC on uncertain → stronger signal when model "surrenders"
   - Lower AUROC on uncertain → uncertain sycophancy is more subtle
   - Same AUROC → one mechanism regardless of uncertainty

Usage:
    python scripts/exp3_probe_by_uncertainty.py
    python scripts/exp3_probe_by_uncertainty.py --layer-fracs 0.5 0.75
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ACTIVATIONS_DIR,
    DEFAULT_LAYER_FRACS,
    GENERATED_DIR,
    NUM_LAYERS,
    RESULTS_DIR,
    TRUNCATION_POINTS,
)
from src.evaluate import bootstrap_auroc_se, compute_auroc
from src.lib import load_activations, mean_pool_at_percent, read_jsonl, resolve_layer_indices
from src.probes.linear import sweep_linear_split


def load_all_activations(
    qids: list[str], layer_indices: list[int],
) -> dict[str, dict[int, np.ndarray]]:
    """Load activations for all question IDs."""
    cache_dir = ACTIVATIONS_DIR / "sycophancy"
    activations = {}
    missing = 0
    for qid in tqdm(qids, desc="loading activations", unit="ex"):
        acts = load_activations(cache_dir / f"{qid}.npz", layer_indices)
        if acts is not None:
            activations[qid] = acts
        else:
            missing += 1
    print(f"Loaded activations: {len(activations)}/{len(qids)} ({missing} missing)", flush=True)
    return activations


def evaluate_on_subset(
    pipeline,
    activations: dict[str, dict[int, np.ndarray]],
    records: list[dict],
    layer_idx: int | str,
    layer_indices: list[int],
    k_pct: float,
) -> dict | None:
    """Evaluate a trained probe on a subset of records.

    Returns dict with auroc, auroc_se, accuracy, n, n_pos, n_neg.
    """
    valid = [r for r in records if r["question_id"] in activations]
    if len(valid) < 10:
        return None

    labels = np.array([int(r["label"]) for r in valid])
    if len(np.unique(labels)) < 2:
        return None

    if layer_idx == "all":
        X = np.stack([
            np.concatenate([
                mean_pool_at_percent(
                    activations[r["question_id"]][l].astype(np.float32), k_pct
                )
                for l in layer_indices
            ])
            for r in valid
        ])
    else:
        X = np.stack([
            mean_pool_at_percent(
                activations[r["question_id"]][layer_idx].astype(np.float32), k_pct
            )
            for r in valid
        ])

    probs = pipeline.predict_proba(X)[:, 1]
    preds = pipeline.predict(X)

    auroc = compute_auroc(labels, probs)
    auroc_se = bootstrap_auroc_se(labels, probs)
    accuracy = float((preds == labels).mean())

    return {
        "auroc": round(auroc, 4),
        "auroc_se": round(auroc_se, 4),
        "accuracy": round(accuracy, 4),
        "n": len(valid),
        "n_pos": int(labels.sum()),
        "n_neg": int((1 - labels).sum()),
        "pos_rate": round(float(labels.mean()), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-fracs", nargs="+", type=float, default=None)
    parser.add_argument("--k-percents", nargs="+", type=float, default=None)
    parser.add_argument("--n-quintiles", type=int, default=5)
    args = parser.parse_args()

    layer_indices = resolve_layer_indices(NUM_LAYERS, args.layer_fracs)
    k_percents = args.k_percents or [float(k) for k in TRUNCATION_POINTS]

    # ─── Load labeled data + splits ──────────────────────────────────────
    labeled_path = GENERATED_DIR / "sycophancy" / "labeled.jsonl"
    splits_path = GENERATED_DIR / "sycophancy" / "splits.json"

    if not labeled_path.exists() or not splits_path.exists():
        print("Run extraction first: python -m src.extract --device cuda")
        print("This creates labeled.jsonl and splits.json")
        return

    records = list(read_jsonl(labeled_path))
    with open(splits_path) as f:
        splits = json.load(f)

    rec_lookup = {r["question_id"]: r for r in records}

    print(f"Loaded {len(records)} labeled records")
    print(f"  Sycophantic: {sum(r['label'] for r in records)} ({100*sum(r['label'] for r in records)/len(records):.1f}%)")

    # ─── Load activations ────────────────────────────────────────────────
    all_qids = [r["question_id"] for r in records]
    activations = load_all_activations(all_qids, layer_indices)

    if len(activations) < 100:
        print("Too few activations. Run extraction first.")
        return

    # ─── Train probes on full dataset ────────────────────────────────────
    print("\n=== Training linear probes (full dataset) ===")
    probe_results = sweep_linear_split(
        activations, records, splits, layer_indices, k_percents,
    )

    # Find best config
    best_key = max(probe_results, key=lambda k: probe_results[k]["test_auroc"])
    best = probe_results[best_key]
    print(f"\nBest config: layer={best_key[0]}, K={best_key[1]}%")
    print(f"  test_auroc={best['test_auroc']:.4f}, acc={best['test_accuracy']:.4f}")

    # ─── Stratify by uncertainty quintiles ────────────────────────────────
    print(f"\n=== Evaluating per uncertainty quintile (N={args.n_quintiles}) ===")

    # Get uncertainty scores
    uncertainties = np.array([
        rec_lookup[qid]["uncertainty_score"]
        for qid in splits["test"]
        if qid in rec_lookup and qid in activations
    ])
    test_qids = [
        qid for qid in splits["test"]
        if qid in rec_lookup and qid in activations
    ]

    quintile_edges = np.percentile(uncertainties, np.linspace(0, 100, args.n_quintiles + 1))
    quintile_labels = np.digitize(uncertainties, quintile_edges[1:-1])  # 0 to n-1

    print(f"\n{'Quintile':<10} {'Range':<20} {'N':<6} {'Pos%':<8} {'AUROC':<12} {'Acc':<8}")
    print("-" * 65)

    quintile_results = {}

    for q_idx in range(args.n_quintiles):
        mask = quintile_labels == q_idx
        q_qids = [qid for qid, m in zip(test_qids, mask) if m]
        q_records = [rec_lookup[qid] for qid in q_qids if qid in rec_lookup]

        # Evaluate best probe on this subset
        best_layer, best_k = best_key
        pipeline = probe_results[best_key]["pipeline"]

        result = evaluate_on_subset(
            pipeline, activations, q_records, best_layer, layer_indices, best_k,
        )

        if result is None:
            print(f"Q{q_idx+1:<9} (too few samples or single class)")
            continue

        lo = uncertainties[mask].min()
        hi = uncertainties[mask].max()
        print(
            f"Q{q_idx+1:<9} {lo:.1f}-{hi:.1f}{'':<12} "
            f"{result['n']:<6} {result['pos_rate']:.1%}{'':<4} "
            f"{result['auroc']:.4f}±{result['auroc_se']:.4f} "
            f"{result['accuracy']:.4f}"
        )

        quintile_results[f"Q{q_idx+1}"] = {
            **result,
            "uncertainty_range": [float(lo), float(hi)],
        }

    # ─── Evaluate all configs per quintile (detailed) ─────────────────────
    print(f"\n=== Detailed: AUROC per (layer, K%) per quintile ===")

    detailed = {}
    for (layer_idx, k_pct), res in sorted(probe_results.items()):
        if not isinstance(layer_idx, int):
            continue  # skip "all" for detailed view

        pipeline = res["pipeline"]
        row = {"overall_test_auroc": res["test_auroc"]}

        for q_idx in range(args.n_quintiles):
            mask = quintile_labels == q_idx
            q_qids = [qid for qid, m in zip(test_qids, mask) if m]
            q_records = [rec_lookup[qid] for qid in q_qids if qid in rec_lookup]

            sub = evaluate_on_subset(
                pipeline, activations, q_records, layer_idx, layer_indices, k_pct,
            )
            if sub:
                row[f"Q{q_idx+1}_auroc"] = sub["auroc"]

        detailed[f"L{layer_idx}_K{k_pct}"] = row

    # Print summary table
    print(f"\n{'Config':<15} {'Overall':<10}", end="")
    for q_idx in range(args.n_quintiles):
        print(f"{'Q'+str(q_idx+1):<10}", end="")
    print()
    print("-" * (15 + 10 + 10 * args.n_quintiles))

    for config_name, row in sorted(detailed.items()):
        print(f"{config_name:<15} {row['overall_test_auroc']:<10.4f}", end="")
        for q_idx in range(args.n_quintiles):
            key = f"Q{q_idx+1}_auroc"
            if key in row:
                print(f"{row[key]:<10.4f}", end="")
            else:
                print(f"{'N/A':<10}", end="")
        print()

    # ─── Save results ─────────────────────────────────────────────────────
    out = {
        "n_records": len(records),
        "n_activations": len(activations),
        "layer_indices": layer_indices,
        "k_percents": k_percents,
        "best_config": {
            "layer": best_key[0],
            "k_pct": best_key[1],
            "test_auroc": best["test_auroc"],
            "test_accuracy": best["test_accuracy"],
        },
        "probe_results": {
            f"{k[0]}|{k[1]}": {kk: vv for kk, vv in v.items() if kk != "pipeline"}
            for k, v in probe_results.items()
        },
        "quintile_results": quintile_results,
        "detailed_by_quintile": detailed,
    }

    out_path = RESULTS_DIR / "exp3_probe_by_uncertainty.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
