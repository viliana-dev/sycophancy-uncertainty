"""Geometric analysis of sycophancy probe vectors.

Trains three probes (overall, confident-only, uncertain-only) and analyses
the geometry of their weight vectors:
  - Cosine similarities between w_conf, w_unc, w_overall
  - Decomposition: w_conf = α·w_unc + v_override
  - PCA of test activations colored by (confidence × sycophancy)
  - Projection onto the (w_unc, v_override) basis

No GPU required — runs on precomputed features.

Usage:
    python scripts/exp_geometry.py
    python scripts/exp_geometry.py --model-family gptoss
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, GENERATED_DIR, PLOTS_DIR, RESULTS_DIR
from src.evaluate import compute_auroc
from src.lib import read_jsonl


BEST_LAYER_QWEN = 30
BEST_LAYER_GPTOSS = 18
BEST_K = 100

GROUP_NAMES = {
    0: "confident_nonsyco",
    1: "confident_syco",
    2: "uncertain_nonsyco",
    3: "uncertain_syco",
}
GROUP_COLORS = {0: "#2196F3", 1: "#F44336", 2: "#64B5F6", 3: "#EF9A9A"}
GROUP_MARKERS = {0: "o", 1: "o", 2: "^", 3: "^"}
GROUP_LABELS = {
    0: "Confident + Non-syco",
    1: "Confident + Syco",
    2: "Uncertain + Non-syco",
    3: "Uncertain + Syco",
}


def load_features(feat_dir, layer: int, k_pct: int) -> tuple[np.ndarray, np.ndarray]:
    npz = np.load(feat_dir / f"L{layer}_K{k_pct}.npz")
    X, qids = npz["X"], npz["qids"]
    npz.close()
    return X, qids


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def train_probe(X_tr, y_tr, C=1.0):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


def extract_weight(pipe) -> np.ndarray:
    """Extract weight vector in scaled feature space."""
    return pipe.named_steps["lr"].coef_[0].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-family", choices=["qwen", "gptoss"], default="qwen")
    parser.add_argument("--metric", choices=["entropy", "heuristic"], default="entropy")
    args = parser.parse_args()

    is_gptoss = args.model_family == "gptoss"
    suffix = "_gptoss" if is_gptoss else ""
    feat_dir = DATA_DIR / "features" / f"sycophancy{suffix}"
    gen_dir = GENERATED_DIR / f"sycophancy{suffix}"
    best_layer = BEST_LAYER_GPTOSS if is_gptoss else BEST_LAYER_QWEN
    model_label = "gpt-oss-20b" if is_gptoss else "Qwen3-14B"

    # ── Load features ────────────────────────────────────────────────────
    X, qids = load_features(feat_dir, best_layer, BEST_K)
    qid_to_idx = {q: i for i, q in enumerate(qids)}
    print(f"Features L{best_layer}_K{BEST_K}: {X.shape} ({model_label})", flush=True)

    # ── Labels & splits ──────────────────────────────────────────────────
    records = list(read_jsonl(gen_dir / "labeled.jsonl"))
    rec_lookup = {r["question_id"]: r for r in records}
    with open(gen_dir / "splits.json") as f:
        splits = json.load(f)

    # ── Uncertainty metric ───────────────────────────────────────────────
    if args.metric == "entropy":
        ent_path = gen_dir / "answer_entropy.jsonl"
        if not ent_path.exists():
            print(f"Missing {ent_path}, falling back to heuristic")
            args.metric = "heuristic"
        else:
            unc_lookup = {}
            for rec in read_jsonl(ent_path):
                unc_lookup[rec["question_id"]] = rec["entropy"]
            print(f"Loaded {len(unc_lookup)} entropy records", flush=True)

    if args.metric == "heuristic":
        unc_lookup = {r["question_id"]: r["uncertainty_score"] for r in records}

    # ── Median split (trainval only) ─────────────────────────────────────
    trainval_qids = [q for q in splits["train"] + splits["val"]
                     if q in qid_to_idx and q in unc_lookup]
    test_qids = [q for q in splits["test"]
                 if q in qid_to_idx and q in unc_lookup]

    tv_unc = np.array([unc_lookup[q] for q in trainval_qids])
    median = float(np.median(tv_unc))
    print(f"Uncertainty median (trainval): {median:.6f}", flush=True)

    def split_by_unc(qid_list):
        conf, unc = [], []
        for q in qid_list:
            if q not in qid_to_idx or q not in unc_lookup:
                continue
            (conf if unc_lookup[q] <= median else unc).append(q)
        return conf, unc

    train_conf, train_unc = split_by_unc(splits["train"] + splits["val"])
    test_conf, test_unc = split_by_unc(splits["test"])

    def to_Xy(qid_list):
        idx = [qid_to_idx[q] for q in qid_list]
        return X[idx], np.array([rec_lookup[q]["label"] for q in qid_list])

    X_tv_all, y_tv_all = to_Xy(trainval_qids)
    X_tv_conf, y_tv_conf = to_Xy(train_conf)
    X_tv_unc, y_tv_unc = to_Xy(train_unc)
    X_te_all, y_te_all = to_Xy(test_qids)

    print(f"Train: {len(trainval_qids)} (conf={len(train_conf)}, unc={len(train_unc)})", flush=True)
    print(f"Test:  {len(test_qids)} (conf={len(test_conf)}, unc={len(test_unc)})", flush=True)

    # ── Train 3 probes ───────────────────────────────────────────────────
    pipe_overall = train_probe(X_tv_all, y_tv_all)
    pipe_conf = train_probe(X_tv_conf, y_tv_conf)
    pipe_unc = train_probe(X_tv_unc, y_tv_unc)

    w_overall = extract_weight(pipe_overall)
    w_conf = extract_weight(pipe_conf)
    w_unc = extract_weight(pipe_unc)

    # Sanity: AUROC of overall probe
    probs_all = pipe_overall.predict_proba(X_te_all)[:, 1]
    auroc_overall = compute_auroc(y_te_all, probs_all)
    print(f"\nOverall probe test AUROC: {auroc_overall:.4f}", flush=True)

    # ══════════════════════════════════════════════════════════════════════
    # A. Cosine similarities
    # ══════════════════════════════════════════════════════════════════════
    cos_conf_unc = cosine_sim(w_conf, w_unc)
    cos_conf_all = cosine_sim(w_conf, w_overall)
    cos_unc_all = cosine_sim(w_unc, w_overall)

    print(f"\n=== Cosine Similarities ===")
    print(f"  cos(w_conf, w_unc)     = {cos_conf_unc:.4f}")
    print(f"  cos(w_conf, w_overall) = {cos_conf_all:.4f}")
    print(f"  cos(w_unc,  w_overall) = {cos_unc_all:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # B. Vector decomposition: w_conf = α·w_unc + v_override
    # ══════════════════════════════════════════════════════════════════════
    alpha = float(np.dot(w_conf, w_unc) / (np.dot(w_unc, w_unc) + 1e-12))
    v_override = w_conf - alpha * w_unc
    orth_check = float(np.dot(v_override, w_unc))

    norm_conf = float(np.linalg.norm(w_conf))
    norm_unc = float(np.linalg.norm(w_unc))
    norm_override = float(np.linalg.norm(v_override))
    norm_proj = float(np.linalg.norm(alpha * w_unc))

    # Fraction of w_conf explained by w_unc
    frac_explained = norm_proj ** 2 / (norm_conf ** 2 + 1e-12)

    print(f"\n=== Decomposition: w_conf = α·w_unc + v_override ===")
    print(f"  α = {alpha:.4f}")
    print(f"  ||w_conf|| = {norm_conf:.4f}")
    print(f"  ||w_unc||  = {norm_unc:.4f}")
    print(f"  ||α·w_unc|| (projection) = {norm_proj:.4f}")
    print(f"  ||v_override||            = {norm_override:.4f}")
    print(f"  Fraction explained by w_unc: {frac_explained:.4f} ({frac_explained*100:.1f}%)")
    print(f"  Orthogonality check: dot(v_override, w_unc) = {orth_check:.2e}")

    # Also decompose w_unc onto w_conf for symmetry
    alpha_rev = float(np.dot(w_unc, w_conf) / (np.dot(w_conf, w_conf) + 1e-12))
    v_residual_rev = w_unc - alpha_rev * w_conf
    norm_proj_rev = float(np.linalg.norm(alpha_rev * w_conf))
    frac_rev = norm_proj_rev ** 2 / (norm_unc ** 2 + 1e-12)
    print(f"\n  Reverse: w_unc = β·w_conf + residual")
    print(f"  β = {alpha_rev:.4f}")
    print(f"  ||β·w_conf|| = {norm_proj_rev:.4f}")
    print(f"  Fraction of w_unc explained by w_conf: {frac_rev:.4f} ({frac_rev*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # C. Assign test records to 4 groups
    # ══════════════════════════════════════════════════════════════════════
    test_groups = np.full(len(test_qids), -1, dtype=int)
    for i, q in enumerate(test_qids):
        confident = unc_lookup[q] <= median
        syco = bool(rec_lookup[q]["label"])
        if confident and not syco:
            test_groups[i] = 0
        elif confident and syco:
            test_groups[i] = 1
        elif not confident and not syco:
            test_groups[i] = 2
        elif not confident and syco:
            test_groups[i] = 3

    # ══════════════════════════════════════════════════════════════════════
    # D. Projection analysis onto (w_unc, v_override) basis
    # ══════════════════════════════════════════════════════════════════════
    # Scale test features with overall scaler for consistent space
    scaler = pipe_overall.named_steps["scaler"]
    X_te_scaled = scaler.transform(X_te_all)

    w_unc_hat = w_unc / (np.linalg.norm(w_unc) + 1e-12)
    v_override_hat = v_override / (np.linalg.norm(v_override) + 1e-12)

    proj_unc = X_te_scaled @ w_unc_hat
    proj_override = X_te_scaled @ v_override_hat

    print(f"\n=== Projection Analysis (mean ± SE per group) ===")
    print(f"{'Group':<24} {'N':<6} {'proj(w_unc)':<18} {'proj(v_override)':<18}")
    print("-" * 70)

    projection_stats = {}
    for g_idx, name in GROUP_NAMES.items():
        mask = test_groups == g_idx
        n = int(mask.sum())
        if n == 0:
            continue
        pu = proj_unc[mask]
        po = proj_override[mask]
        pu_mean, pu_se = float(pu.mean()), float(pu.std(ddof=1) / np.sqrt(n))
        po_mean, po_se = float(po.mean()), float(po.std(ddof=1) / np.sqrt(n))
        print(f"{name:<24} {n:<6} {pu_mean:+.4f}±{pu_se:.4f}   {po_mean:+.4f}±{po_se:.4f}")
        projection_stats[name] = {
            "n": n, "proj_w_unc_mean": round(pu_mean, 4),
            "proj_w_unc_se": round(pu_se, 4),
            "proj_v_override_mean": round(po_mean, 4),
            "proj_v_override_se": round(po_se, 4),
        }

    # Cohen's d between syco and nonsyco within each regime
    print(f"\n=== Effect Sizes (Cohen's d, syco vs nonsyco) ===")
    effect_sizes = {}
    for regime, g_nonsyco, g_syco in [("confident", 0, 1), ("uncertain", 2, 3)]:
        m_ns = test_groups == g_nonsyco
        m_s = test_groups == g_syco
        if m_ns.sum() < 5 or m_s.sum() < 5:
            continue
        for axis_name, proj_vals in [("w_unc", proj_unc), ("v_override", proj_override)]:
            v_ns = proj_vals[m_ns]
            v_s = proj_vals[m_s]
            pooled_std = np.sqrt((v_ns.var(ddof=1) * (len(v_ns) - 1) +
                                  v_s.var(ddof=1) * (len(v_s) - 1)) /
                                 (len(v_ns) + len(v_s) - 2))
            d = float((v_s.mean() - v_ns.mean()) / (pooled_std + 1e-12))
            t_stat, p_val = stats.ttest_ind(v_s, v_ns)
            print(f"  {regime:<10} {axis_name:<14} d={d:+.3f}  t={t_stat:+.3f}  p={p_val:.2e}")
            effect_sizes[f"{regime}_{axis_name}"] = {
                "cohens_d": round(d, 4), "t_stat": round(float(t_stat), 4),
                "p_value": float(p_val),
            }

    # ══════════════════════════════════════════════════════════════════════
    # E. PCA visualization
    # ══════════════════════════════════════════════════════════════════════
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: PCA of activation space
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_te_scaled)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for g_idx in GROUP_NAMES:
        mask = test_groups == g_idx
        if mask.sum() == 0:
            continue
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=GROUP_COLORS[g_idx], marker=GROUP_MARKERS[g_idx],
                   alpha=0.35, s=20, label=GROUP_LABELS[g_idx])

    # Project probe directions into PCA space
    for w, label, color, ls in [
        (w_conf, "$w_{conf}$", "#C62828", "-"),
        (w_unc, "$w_{unc}$", "#1565C0", "--"),
    ]:
        w_pca = pca.transform(w.reshape(1, -1))[0]
        w_pca = w_pca / (np.linalg.norm(w_pca) + 1e-12)
        scale = max(X_pca[:, 0].std(), X_pca[:, 1].std()) * 2
        ax.annotate("", xy=(w_pca[0] * scale, w_pca[1] * scale),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=color, lw=2, ls=ls))
        ax.text(w_pca[0] * scale * 1.1, w_pca[1] * scale * 1.1, label,
                fontsize=12, color=color, fontweight="bold")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Activation PCA — {model_label} (L{best_layer} K{BEST_K})")
    ax.legend(fontsize=9, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    pca_path = PLOTS_DIR / f"geometry_pca{'_gptoss' if is_gptoss else ''}.png"
    fig.savefig(pca_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {pca_path}", flush=True)

    # Figure 2: Probe space (w_unc × v_override)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for g_idx in GROUP_NAMES:
        mask = test_groups == g_idx
        if mask.sum() == 0:
            continue
        ax.scatter(proj_unc[mask], proj_override[mask],
                   c=GROUP_COLORS[g_idx], marker=GROUP_MARKERS[g_idx],
                   alpha=0.35, s=20, label=GROUP_LABELS[g_idx])

    # Add group means as large markers
    for g_idx in GROUP_NAMES:
        mask = test_groups == g_idx
        if mask.sum() == 0:
            continue
        mx = float(proj_unc[mask].mean())
        my = float(proj_override[mask].mean())
        ax.scatter([mx], [my], c=GROUP_COLORS[g_idx],
                   marker=GROUP_MARKERS[g_idx], s=200, edgecolors="black",
                   linewidths=1.5, zorder=10)

    ax.set_xlabel("Projection onto $\\hat{w}_{unc}$ (shared drift)")
    ax.set_ylabel("Projection onto $\\hat{v}_{override}$ (confident-only)")
    ax.set_title(f"Probe Space — {model_label}")
    ax.legend(fontsize=9, loc="best")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axvline(0, color="gray", lw=0.5, ls=":")
    fig.tight_layout()
    probe_path = PLOTS_DIR / f"geometry_probe_space{'_gptoss' if is_gptoss else ''}.png"
    fig.savefig(probe_path, dpi=150)
    plt.close(fig)
    print(f"Saved {probe_path}", flush=True)

    # Figure 3: Side-by-side group mean bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    group_order = [0, 1, 2, 3]
    labels = [GROUP_LABELS[g] for g in group_order]
    colors = [GROUP_COLORS[g] for g in group_order]

    for ax_i, (axis_name, proj_vals) in enumerate([
        ("$\\hat{w}_{unc}$", proj_unc), ("$\\hat{v}_{override}$", proj_override)
    ]):
        means, ses = [], []
        for g in group_order:
            mask = test_groups == g
            v = proj_vals[mask]
            means.append(float(v.mean()) if mask.sum() > 0 else 0)
            ses.append(float(v.std(ddof=1) / np.sqrt(mask.sum())) if mask.sum() > 1 else 0)
        axes[ax_i].bar(range(4), means, yerr=ses, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
        axes[ax_i].set_xticks(range(4))
        axes[ax_i].set_xticklabels(["C+NS", "C+S", "U+NS", "U+S"], fontsize=10)
        axes[ax_i].set_ylabel(f"Mean projection onto {axis_name}")
        axes[ax_i].axhline(0, color="gray", lw=0.5, ls=":")
        axes[ax_i].set_title(axis_name)

    fig.suptitle(f"Group Projections — {model_label}", fontsize=13)
    fig.tight_layout()
    bar_path = PLOTS_DIR / f"geometry_bars{'_gptoss' if is_gptoss else ''}.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"Saved {bar_path}", flush=True)

    # ── Save JSON results ────────────────────────────────────────────────
    out = {
        "model": model_label,
        "layer": best_layer,
        "k_pct": BEST_K,
        "metric": args.metric,
        "median": median,
        "n_train_conf": len(train_conf),
        "n_train_unc": len(train_unc),
        "n_test": len(test_qids),
        "overall_auroc": round(auroc_overall, 4),
        "cosine_similarities": {
            "conf_unc": round(cos_conf_unc, 4),
            "conf_overall": round(cos_conf_all, 4),
            "unc_overall": round(cos_unc_all, 4),
        },
        "decomposition": {
            "alpha": round(alpha, 4),
            "norm_w_conf": round(norm_conf, 4),
            "norm_w_unc": round(norm_unc, 4),
            "norm_projection": round(norm_proj, 4),
            "norm_v_override": round(norm_override, 4),
            "fraction_explained": round(frac_explained, 4),
            "orthogonality_check": orth_check,
            "reverse_beta": round(alpha_rev, 4),
            "reverse_fraction_explained": round(frac_rev, 4),
        },
        "projection_stats": projection_stats,
        "effect_sizes": effect_sizes,
        "pca_variance_explained": [round(float(v), 4) for v in pca.explained_variance_ratio_[:2]],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"exp_geometry{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
