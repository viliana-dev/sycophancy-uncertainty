"""Subspace analysis: Linear CKA, principal angles, inclusion scores, projection residuals.

GPU-accelerated (torch) — all heavy linear algebra on CUDA.

Quantifies the geometric relationship between confident and uncertain sycophancy
subspaces across all three architectures. Strengthens the "nested subspace" claim
(unc ⊂ conf).

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/exp_subspace_analysis.py
    CUDA_VISIBLE_DEVICES=4 python scripts/exp_subspace_analysis.py --model-family qwen
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, GENERATED_DIR, RESULTS_DIR
from src.lib import read_jsonl

MODELS = {
    "qwen": {"layer": 30, "label": "Qwen3-14B"},
    "gptoss": {"layer": 18, "label": "gpt-oss-20b"},
    "qwen_moe": {"layer": 36, "label": "Qwen3-30B-A3B"},
}
BEST_K = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Data loading (mirrors exp_geometry.py) ──────────────────────────────────

def load_features(feat_dir, layer: int, k_pct: int):
    npz = np.load(feat_dir / f"L{layer}_K{k_pct}.npz")
    X, qids = npz["X"], npz["qids"]
    npz.close()
    return X, qids


def load_data(family: str):
    suffix = "" if family == "qwen" else f"_{family}"
    cfg = MODELS[family]
    feat_dir = DATA_DIR / "features" / f"sycophancy{suffix}"
    gen_dir = GENERATED_DIR / f"sycophancy{suffix}"

    X, qids = load_features(feat_dir, cfg["layer"], BEST_K)
    qid_to_idx = {q: i for i, q in enumerate(qids)}

    records = list(read_jsonl(gen_dir / "labeled.jsonl"))
    rec_lookup = {r["question_id"]: r for r in records}
    with open(gen_dir / "splits.json") as f:
        splits = json.load(f)

    unc_lookup = {}
    for rec in read_jsonl(gen_dir / "answer_entropy.jsonl"):
        unc_lookup[rec["question_id"]] = rec["entropy"]

    trainval_qids = [q for q in splits["train"] + splits["val"]
                     if q in qid_to_idx and q in unc_lookup]
    test_qids = [q for q in splits["test"]
                 if q in qid_to_idx and q in unc_lookup]

    tv_unc = np.array([unc_lookup[q] for q in trainval_qids])
    median = float(np.median(tv_unc))

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

    return {
        "X_tv_conf": to_Xy(train_conf),
        "X_tv_unc": to_Xy(train_unc),
        "X_te_conf": to_Xy(test_conf),
        "X_te_unc": to_Xy(test_unc),
        "X_tv_all": to_Xy(trainval_qids),
        "X_te_all": to_Xy(test_qids),
        "median": median,
        "n_train_conf": len(train_conf),
        "n_train_unc": len(train_unc),
        "n_test_conf": len(test_conf),
        "n_test_unc": len(test_unc),
    }


# ─── Probe training (CPU — sklearn) ─────────────────────────────────────────

def train_probe(X_tr, y_tr, C=1.0):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


# ─── GPU-accelerated linear algebra ─────────────────────────────────────────

def to_gpu(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(DEVICE)
    return x.float().to(DEVICE)


@torch.no_grad()
def linear_cka_gpu(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA on GPU. X: (n, p), Y: (n, q)."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    XtY = X.T @ Y
    hsic_xy = (XtY ** 2).sum()
    hsic_xx = ((X.T @ X) ** 2).sum()
    hsic_yy = ((Y.T @ Y) ** 2).sum()

    return float(hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-12))


@torch.no_grad()
def principal_angles_gpu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Principal angles between column spaces. Returns angles in radians."""
    Q_A = torch.linalg.qr(A).Q
    Q_B = torch.linalg.qr(B).Q
    sigmas = torch.linalg.svdvals(Q_A.T @ Q_B)
    sigmas = sigmas.clamp(-1.0, 1.0)
    return torch.arccos(sigmas)


@torch.no_grad()
def subspace_inclusion_gpu(V_sub: torch.Tensor, V_super: torch.Tensor) -> float:
    """Fraction of V_sub captured by V_super. Asymmetric when dims differ."""
    Q_sub = torch.linalg.qr(V_sub).Q
    Q_super = torch.linalg.qr(V_super).Q
    proj = Q_super @ (Q_super.T @ Q_sub)
    return float((proj ** 2).sum(dim=0).mean())


@torch.no_grad()
def projection_residual_gpu(X: torch.Tensor, V_from: torch.Tensor, V_onto: torch.Tensor) -> float:
    """Fraction of V_from-projected variance lost when projecting onto V_onto."""
    Q_from = torch.linalg.qr(V_from).Q
    Q_onto = torch.linalg.qr(V_onto).Q

    X_from = X @ Q_from @ Q_from.T
    X_from_onto = X_from @ Q_onto @ Q_onto.T

    norm_from = (X_from ** 2).sum()
    norm_residual = ((X_from - X_from_onto) ** 2).sum()
    return float(norm_residual / (norm_from + 1e-12))


@torch.no_grad()
def build_discriminant_subspace_gpu(X_sc: torch.Tensor, y: torch.Tensor, n_components: int):
    """Build discriminant subspace on GPU via truncated SVD of within-class scatter."""
    mask_pos = y == 1
    mask_neg = y == 0

    mu_pos = X_sc[mask_pos].mean(dim=0) if mask_pos.sum() > 0 else X_sc.mean(dim=0)
    mu_neg = X_sc[mask_neg].mean(dim=0) if mask_neg.sum() > 0 else X_sc.mean(dim=0)

    delta = mu_pos - mu_neg
    d1 = delta / (delta.norm() + 1e-12)

    if n_components == 1:
        return d1.unsqueeze(1)

    # Within-class centered data
    parts = []
    if mask_pos.sum() > 1:
        parts.append(X_sc[mask_pos] - X_sc[mask_pos].mean(dim=0))
    if mask_neg.sum() > 1:
        parts.append(X_sc[mask_neg] - X_sc[mask_neg].mean(dim=0))
    X_within = torch.cat(parts, dim=0) if parts else X_sc - X_sc.mean(dim=0)

    # Truncated SVD — only need top n_components*2
    k = min(n_components * 2, X_within.shape[0], X_within.shape[1])
    U, S, Vt = torch.svd_lowrank(X_within, q=k)
    candidates = Vt  # (d, k)

    # Gram-Schmidt: d1 first, then orthogonalize SVD directions
    basis = [d1]
    for i in range(candidates.shape[1]):
        v = candidates[:, i].clone()
        for b in basis:
            v = v - torch.dot(v, b) * b
        nv = v.norm()
        if nv > 1e-8:
            basis.append(v / nv)
        if len(basis) >= n_components:
            break

    # Pad if needed
    d = X_sc.shape[1]
    while len(basis) < n_components:
        v = torch.randn(d, device=DEVICE)
        for b in basis:
            v = v - torch.dot(v, b) * b
        nv = v.norm()
        if nv > 1e-8:
            basis.append(v / nv)

    return torch.stack(basis, dim=1)  # (d, n_components)


# ─── Main analysis ───────────────────────────────────────────────────────────

def analyze_model(family: str) -> dict:
    cfg = MODELS[family]
    print(f"\n{'='*60}")
    print(f"  {cfg['label']} (L{cfg['layer']} K{BEST_K}) — device={DEVICE}")
    print(f"{'='*60}", flush=True)

    data = load_data(family)
    X_tv_conf_np, y_tv_conf_np = data["X_tv_conf"]
    X_tv_unc_np, y_tv_unc_np = data["X_tv_unc"]
    X_te_conf_np, y_te_conf_np = data["X_te_conf"]
    X_te_unc_np, y_te_unc_np = data["X_te_unc"]
    X_tv_all_np, y_tv_all_np = data["X_tv_all"]
    X_te_all_np, y_te_all_np = data["X_te_all"]

    print(f"Train: conf={len(y_tv_conf_np)}, unc={len(y_tv_unc_np)}")
    print(f"Test:  conf={len(y_te_conf_np)}, unc={len(y_te_unc_np)}", flush=True)

    # Train probes (CPU)
    pipe_conf = train_probe(X_tv_conf_np, y_tv_conf_np)
    pipe_unc = train_probe(X_tv_unc_np, y_tv_unc_np)
    pipe_all = train_probe(X_tv_all_np, y_tv_all_np)
    scaler = pipe_all.named_steps["scaler"]

    # Scale all data and move to GPU
    X_tv_conf_sc = to_gpu(scaler.transform(X_tv_conf_np))
    X_tv_unc_sc = to_gpu(scaler.transform(X_tv_unc_np))
    X_te_conf_sc = to_gpu(scaler.transform(X_te_conf_np))
    X_te_unc_sc = to_gpu(scaler.transform(X_te_unc_np))
    X_te_all_sc = to_gpu(scaler.transform(np.vstack([X_te_conf_np, X_te_unc_np])))

    y_tv_conf = to_gpu(torch.from_numpy(y_tv_conf_np).long())
    y_tv_unc = to_gpu(torch.from_numpy(y_tv_unc_np).long())

    # Probe directions
    w_conf = pipe_conf.named_steps["lr"].coef_[0].copy()
    w_conf = w_conf / (np.linalg.norm(w_conf) + 1e-12)
    w_unc = pipe_unc.named_steps["lr"].coef_[0].copy()
    w_unc = w_unc / (np.linalg.norm(w_unc) + 1e-12)
    w_conf_gpu = to_gpu(w_conf)
    w_unc_gpu = to_gpu(w_unc)

    results = {
        "model": cfg["label"],
        "layer": cfg["layer"],
        "n_train_conf": data["n_train_conf"],
        "n_train_unc": data["n_train_unc"],
        "n_test_conf": data["n_test_conf"],
        "n_test_unc": data["n_test_unc"],
        "median": data["median"],
    }

    gen = torch.Generator(device=DEVICE).manual_seed(42)

    # ══════════════════════════════════════════════════════════════════════
    # 1. Linear CKA
    # ══════════════════════════════════════════════════════════════════════
    n_min = min(len(X_te_conf_sc), len(X_te_unc_sc))
    idx_conf = torch.randperm(len(X_te_conf_sc), generator=gen, device=DEVICE)[:n_min]
    idx_unc = torch.randperm(len(X_te_unc_sc), generator=gen, device=DEVICE)[:n_min]

    cka_val = linear_cka_gpu(X_te_conf_sc[idx_conf], X_te_unc_sc[idx_unc])
    print(f"\n--- Linear CKA ---")
    print(f"  CKA(conf, unc) = {cka_val:.4f}", flush=True)
    results["linear_cka"] = round(cka_val, 4)

    # CKA sycophantic only
    syco_conf = X_te_conf_sc[to_gpu(torch.from_numpy(y_te_conf_np)).bool()]
    syco_unc = X_te_unc_sc[to_gpu(torch.from_numpy(y_te_unc_np)).bool()]
    if len(syco_conf) > 5 and len(syco_unc) > 5:
        n_s = min(len(syco_conf), len(syco_unc))
        cka_syco = linear_cka_gpu(
            syco_conf[torch.randperm(len(syco_conf), generator=gen, device=DEVICE)[:n_s]],
            syco_unc[torch.randperm(len(syco_unc), generator=gen, device=DEVICE)[:n_s]])
        print(f"  CKA(conf syco, unc syco) = {cka_syco:.4f}")
        results["linear_cka_syco_only"] = round(cka_syco, 4)

    # Bootstrap CI (fast on GPU)
    cka_boots = []
    for _ in range(500):
        idx = torch.randint(n_min, (n_min,), generator=gen, device=DEVICE)
        cka_boots.append(linear_cka_gpu(X_te_conf_sc[idx_conf[idx]], X_te_unc_sc[idx_unc[idx]]))
    cka_ci = (np.percentile(cka_boots, 2.5), np.percentile(cka_boots, 97.5))
    print(f"  95% CI: [{cka_ci[0]:.4f}, {cka_ci[1]:.4f}]")
    results["linear_cka_ci95"] = [round(float(cka_ci[0]), 4), round(float(cka_ci[1]), 4)]

    # ══════════════════════════════════════════════════════════════════════
    # 2. Probe direction angle
    # ══════════════════════════════════════════════════════════════════════
    cos_1d = float(torch.dot(w_conf_gpu, w_unc_gpu))
    angle_1d = float(np.degrees(np.arccos(np.clip(abs(cos_1d), 0, 1))))
    print(f"\n--- Probe direction angle ---")
    print(f"  cos(w_conf, w_unc) = {cos_1d:.4f}")
    print(f"  angle = {angle_1d:.1f}°", flush=True)
    results["probe_cosine"] = round(cos_1d, 4)
    results["probe_angle_deg"] = round(angle_1d, 2)

    # ══════════════════════════════════════════════════════════════════════
    # 3. Build discriminant subspaces (GPU)
    # ══════════════════════════════════════════════════════════════════════
    subspaces = {}
    for k in [3, 5, 10, 20]:
        subspaces[("conf", k)] = build_discriminant_subspace_gpu(X_tv_conf_sc, y_tv_conf, k)
        subspaces[("unc", k)] = build_discriminant_subspace_gpu(X_tv_unc_sc, y_tv_unc, k)
    print(f"\nBuilt discriminant subspaces (dims: 3, 5, 10, 20)", flush=True)

    # ══════════════════════════════════════════════════════════════════════
    # 4. Principal angles (equal dim)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n--- Principal angles (equal dim) ---")
    for k in [3, 5, 10]:
        angles = principal_angles_gpu(subspaces[("conf", k)], subspaces[("unc", k)])
        angles_deg = angles.cpu().numpy() * 180 / np.pi
        print(f"  k={k}: {', '.join(f'{a:.1f}°' for a in angles_deg)}")
        print(f"        min={angles_deg[0]:.1f}°  max={angles_deg[-1]:.1f}°  mean={angles_deg.mean():.1f}°")
        results[f"principal_angles_k{k}"] = {
            "angles_deg": [round(float(a), 2) for a in angles_deg],
            "min_deg": round(float(angles_deg[0]), 2),
            "max_deg": round(float(angles_deg[-1]), 2),
            "mean_deg": round(float(angles_deg.mean()), 2),
        }

    # ══════════════════════════════════════════════════════════════════════
    # 5. Asymmetric subspace inclusion
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n--- Asymmetric subspace inclusion ---")
    for k_sub, k_super in [(3, 10), (5, 10), (5, 20), (3, 20)]:
        inc_unc_in_conf = subspace_inclusion_gpu(subspaces[("unc", k_sub)], subspaces[("conf", k_super)])
        inc_conf_in_unc = subspace_inclusion_gpu(subspaces[("conf", k_sub)], subspaces[("unc", k_super)])
        asym = inc_unc_in_conf - inc_conf_in_unc
        print(f"  sub={k_sub}, super={k_super}: "
              f"inc(unc⊂conf)={inc_unc_in_conf:.4f}  "
              f"inc(conf⊂unc)={inc_conf_in_unc:.4f}  "
              f"Δ={asym:+.4f}")
        results[f"inclusion_sub{k_sub}_super{k_super}"] = {
            "unc_in_conf": round(inc_unc_in_conf, 4),
            "conf_in_unc": round(inc_conf_in_unc, 4),
            "asymmetry": round(asym, 4),
        }

    # Equal-dim sanity check
    print(f"\n--- Equal-dim inclusion (sanity: should be symmetric) ---")
    for k in [5, 10]:
        inc_a = subspace_inclusion_gpu(subspaces[("unc", k)], subspaces[("conf", k)])
        inc_b = subspace_inclusion_gpu(subspaces[("conf", k)], subspaces[("unc", k)])
        print(f"  k={k}: inc(unc⊂conf)={inc_a:.4f}  inc(conf⊂unc)={inc_b:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # 6. Projection residuals
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n--- Projection residuals ---")
    for k_from, k_onto in [(3, 10), (5, 10), (5, 20)]:
        pr_unc = projection_residual_gpu(X_te_all_sc, subspaces[("unc", k_from)], subspaces[("conf", k_onto)])
        pr_conf = projection_residual_gpu(X_te_all_sc, subspaces[("conf", k_from)], subspaces[("unc", k_onto)])
        print(f"  from={k_from}, onto={k_onto}: "
              f"resid(unc→conf)={pr_unc:.4f}  resid(conf→unc)={pr_conf:.4f}  "
              f"Δ={pr_conf - pr_unc:+.4f}")
        results[f"proj_resid_from{k_from}_onto{k_onto}"] = {
            "unc_onto_conf": round(pr_unc, 4),
            "conf_onto_unc": round(pr_conf, 4),
            "asymmetry": round(pr_conf - pr_unc, 4),
        }

    # Equal-dim
    print(f"\n--- Equal-dim projection residuals ---")
    for k in [5, 10, 20]:
        pr_a = projection_residual_gpu(X_te_all_sc, subspaces[("unc", k)], subspaces[("conf", k)])
        pr_b = projection_residual_gpu(X_te_all_sc, subspaces[("conf", k)], subspaces[("unc", k)])
        print(f"  k={k}: resid(unc→conf)={pr_a:.4f}  resid(conf→unc)={pr_b:.4f}  Δ={pr_b - pr_a:+.4f}")
        results[f"proj_resid_equal_k{k}"] = {
            "unc_onto_conf": round(pr_a, 4),
            "conf_onto_unc": round(pr_b, 4),
            "asymmetry": round(pr_b - pr_a, 4),
        }

    # ══════════════════════════════════════════════════════════════════════
    # 7. Permutation test (GPU — fast)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n--- Permutation test (500 perms, k_sub=5, k_super=20) ---", flush=True)
    k_sub, k_super = 5, 20
    obs_inc_unc = subspace_inclusion_gpu(subspaces[("unc", k_sub)], subspaces[("conf", k_super)])
    obs_inc_conf = subspace_inclusion_gpu(subspaces[("conf", k_sub)], subspaces[("unc", k_super)])
    obs_asym = obs_inc_unc - obs_inc_conf

    X_pool = torch.cat([X_tv_conf_sc, X_tv_unc_sc], dim=0)
    y_pool = torch.cat([y_tv_conf, y_tv_unc], dim=0)
    n_conf = len(y_tv_conf)
    n_perms = 500

    null_asyms = []
    for i in range(n_perms):
        perm = torch.randperm(len(X_pool), generator=gen, device=DEVICE)
        X_a, y_a = X_pool[perm[:n_conf]], y_pool[perm[:n_conf]]
        X_b, y_b = X_pool[perm[n_conf:]], y_pool[perm[n_conf:]]

        V_a_sub = build_discriminant_subspace_gpu(X_a, y_a, k_sub)
        V_b_sup = build_discriminant_subspace_gpu(X_b, y_b, k_super)
        V_b_sub = build_discriminant_subspace_gpu(X_b, y_b, k_sub)
        V_a_sup = build_discriminant_subspace_gpu(X_a, y_a, k_super)

        inc_ab = subspace_inclusion_gpu(V_a_sub, V_b_sup)
        inc_ba = subspace_inclusion_gpu(V_b_sub, V_a_sup)
        null_asyms.append(inc_ab - inc_ba)

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_perms} done", flush=True)

    null_asyms = np.array(null_asyms)
    p_value = float(np.mean(np.abs(null_asyms) >= abs(obs_asym)))
    print(f"  Observed: {obs_asym:+.4f}")
    print(f"  Null: mean={null_asyms.mean():.4f}, std={null_asyms.std():.4f}")
    print(f"  p-value (two-sided): {p_value:.4f}", flush=True)

    results["permutation_test"] = {
        "k_sub": k_sub, "k_super": k_super,
        "observed_asymmetry": round(obs_asym, 4),
        "null_mean": round(float(null_asyms.mean()), 4),
        "null_std": round(float(null_asyms.std()), 4),
        "p_value": round(p_value, 4),
        "n_permutations": n_perms,
    }

    # ══════════════════════════════════════════════════════════════════════
    # 8. Probe direction capture
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n--- Probe direction capture ---")
    for k in [5, 10, 20]:
        Q_conf = torch.linalg.qr(subspaces[("conf", k)]).Q
        Q_unc = torch.linalg.qr(subspaces[("unc", k)]).Q

        proj_wunc_on_conf = Q_conf @ (Q_conf.T @ w_unc_gpu)
        capture_wunc = float((proj_wunc_on_conf ** 2).sum())

        proj_wconf_on_unc = Q_unc @ (Q_unc.T @ w_conf_gpu)
        capture_wconf = float((proj_wconf_on_unc ** 2).sum())

        print(f"  k={k}: capture(w_unc by conf)={capture_wunc:.4f}  "
              f"capture(w_conf by unc)={capture_wconf:.4f}  "
              f"Δ={capture_wunc - capture_wconf:+.4f}")

        results[f"probe_capture_k{k}"] = {
            "w_unc_by_conf_subspace": round(capture_wunc, 4),
            "w_conf_by_unc_subspace": round(capture_wconf, 4),
            "asymmetry": round(capture_wunc - capture_wconf, 4),
        }

    # v_override capture
    alpha_proj = float(torch.dot(w_conf_gpu, w_unc_gpu) / (torch.dot(w_unc_gpu, w_unc_gpu) + 1e-12))
    v_override = w_conf_gpu - alpha_proj * w_unc_gpu
    v_override = v_override / (v_override.norm() + 1e-12)

    for k in [10, 20]:
        Q_unc_k = torch.linalg.qr(subspaces[("unc", k)]).Q
        proj_vo = Q_unc_k @ (Q_unc_k.T @ v_override)
        capture_vo = float((proj_vo ** 2).sum())
        print(f"  k={k}: capture(v_override by unc)={capture_vo:.4f} (should be low)")
        results[f"v_override_capture_by_unc_k{k}"] = round(capture_vo, 4)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-family", choices=["qwen", "gptoss", "qwen_moe", "all"], default="all")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    families = list(MODELS.keys()) if args.model_family == "all" else [args.model_family]
    all_results = {}

    for family in families:
        try:
            res = analyze_model(family)
            all_results[family] = res

            suffix = "" if family == "qwen" else f"_{family}"
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = RESULTS_DIR / f"exp_subspace_analysis{suffix}.json"
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"\nSaved {out_path}")
        except Exception as e:
            print(f"\nERROR on {family}: {e}")
            import traceback; traceback.print_exc()

    # Cross-model summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("  CROSS-MODEL SUMMARY")
        print(f"{'='*70}")

        fams = [f for f in families if f in all_results]
        header = f"{'Metric':<40}" + "".join(f" {MODELS[f]['label']:>14}" for f in fams)
        print(header)
        print("-" * len(header))

        for key, label in [
            ("linear_cka", "Linear CKA"),
            ("probe_cosine", "cos(w_conf, w_unc)"),
            ("probe_angle_deg", "Probe angle (°)"),
        ]:
            row = f"{label:<40}"
            for f in fams:
                v = all_results[f].get(key, "N/A")
                row += f" {v:>14}"
            print(row)

        print()
        for k_sub, k_super in [(5, 20)]:
            key = f"inclusion_sub{k_sub}_super{k_super}"
            for direction, label in [
                ("unc_in_conf", f"inc(unc⊂conf) {k_sub}→{k_super}"),
                ("conf_in_unc", f"inc(conf⊂unc) {k_sub}→{k_super}"),
                ("asymmetry", f"Δ inclusion {k_sub}→{k_super}"),
            ]:
                row = f"{label:<40}"
                for f in fams:
                    v = all_results[f].get(key, {}).get(direction, "N/A")
                    row += f" {v:>14}"
                print(row)

        print()
        for f in fams:
            pt = all_results[f].get("permutation_test", {})
            print(f"  {MODELS[f]['label']}: perm p={pt.get('p_value', 'N/A')}, "
                  f"obs_Δ={pt.get('observed_asymmetry', 'N/A')}")

        combined_path = RESULTS_DIR / "exp_subspace_analysis_combined.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved {combined_path}")


if __name__ == "__main__":
    main()
