"""Activation steering: subtract sycophancy direction from residual stream during generation.

Tests whether single-direction interventions can reduce sycophancy, and whether
the effect differs between confident and uncertain regimes.

Designed for parallel execution across GPUs — each process handles one direction.

Usage (4-GPU parallel):
    CUDA_VISIBLE_DEVICES=4 python scripts/exp_steering.py --direction baseline --alphas 0 --direction2 w_overall --alphas2 5
    CUDA_VISIBLE_DEVICES=5 python scripts/exp_steering.py --direction w_overall --alphas 10 20
    CUDA_VISIBLE_DEVICES=6 python scripts/exp_steering.py --direction v_override --alphas 5 10
    CUDA_VISIBLE_DEVICES=7 python scripts/exp_steering.py --direction v_override --alphas 20 --direction2 random --alphas2 10
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ANTHROPIC_EVALS_DIR, DATA_DIR, GENERATED_DIR, PLOTS_DIR, RESULTS_DIR,
    SUBJECT_MODEL,
)
from src.lib import (
    apply_chat_template,
    load_model,
    read_jsonl,
    resolve_device,
)
from src.prompts import parse_ab_answer

BEST_LAYER = 30
BEST_K = 100
STEERING_LAYER = 30


# ── Probe training ──────────────────────────────────────────────────────────

def load_features(feat_dir, layer: int, k_pct: int):
    npz = np.load(feat_dir / f"L{layer}_K{k_pct}.npz")
    X, qids = npz["X"], npz["qids"]
    npz.close()
    return X, qids


def train_probe(X_tr, y_tr, C=1.0):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


def probe_to_raw_direction(pipe) -> np.ndarray:
    """Convert probe weight vector from scaled space to raw activation space."""
    w_scaled = pipe.named_steps["lr"].coef_[0]
    scale = pipe.named_steps["scaler"].scale_
    w_raw = w_scaled / scale
    w_raw = w_raw / (np.linalg.norm(w_raw) + 1e-12)
    return w_raw


# ── Steering hook ───────────────────────────────────────────────────────────

def make_steering_hook(direction_tensor, alpha):
    """Forward hook that subtracts alpha * direction from hidden states."""
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # direction_tensor is (hidden_dim,), broadcast across (batch, seq, hidden)
        modified = hidden - alpha * direction_tensor
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    return hook_fn


# ── Generation ──────────────────────────────────────────────────────────────

def parse_thinking_answer(text: str) -> tuple[str, str]:
    """Split model output into (thinking, answer)."""
    if "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0]
        if thinking.startswith("<think>"):
            thinking = thinking[len("<think>"):]
        return thinking.strip(), parts[1].strip()
    if "<think>" in text:
        thinking = text[text.index("<think>") + len("<think>"):]
        return thinking.strip(), ""
    return "", text.strip()


def generate_steered(model, tokenizer, prompt, device,
                     direction_tensor=None, alpha=0.0,
                     max_new_tokens=8192):
    """Generate a response with optional activation steering at STEERING_LAYER."""
    text = apply_chat_template(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    hook = None
    if direction_tensor is not None and alpha != 0:
        hook = model.model.layers[STEERING_LAYER].register_forward_hook(
            make_steering_hook(direction_tensor, alpha)
        )

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if hook is not None:
            hook.remove()

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
    thinking, answer = parse_thinking_answer(raw)
    return answer, thinking


# ── Main ────────────────────────────────────────────────────────────────────

def compute_directions(feat_dir, gen_dir):
    """Train probes and compute steering directions in raw activation space."""
    X, qids = load_features(feat_dir, BEST_LAYER, BEST_K)
    qid_to_idx = {q: i for i, q in enumerate(qids)}

    records = list(read_jsonl(gen_dir / "labeled.jsonl"))
    rec_lookup = {r["question_id"]: r for r in records}
    with open(gen_dir / "splits.json") as f:
        splits = json.load(f)

    # Load answer_matching_behavior from source files
    amb_lookup = {}
    for fname in ANTHROPIC_EVALS_DIR.glob("sycophancy_on_*.jsonl"):
        source_name = fname.stem
        for i, line in enumerate(open(fname)):
            rec = json.loads(line.strip())
            qid = f"{source_name}_{i:05d}"
            amb_lookup[qid] = rec["answer_matching_behavior"]

    # Entropy
    ent_path = gen_dir / "answer_entropy.jsonl"
    if ent_path.exists():
        unc_lookup = {}
        for rec in read_jsonl(ent_path):
            unc_lookup[rec["question_id"]] = rec["entropy"]
    else:
        unc_lookup = {r["question_id"]: r["uncertainty_score"] for r in records}

    # Median (trainval)
    trainval_qids = [q for q in splits["train"] + splits["val"]
                     if q in qid_to_idx and q in unc_lookup]
    median = float(np.median([unc_lookup[q] for q in trainval_qids]))

    # Split trainval by confidence
    train_conf, train_unc = [], []
    for q in trainval_qids:
        (train_conf if unc_lookup[q] <= median else train_unc).append(q)

    def to_Xy(qid_list):
        idx = [qid_to_idx[q] for q in qid_list if q in qid_to_idx]
        return X[idx], np.array([rec_lookup[qid_list[i]]["label"] for i in range(len(idx))])

    X_all, y_all = to_Xy(trainval_qids)
    X_conf, y_conf = to_Xy(train_conf)
    X_unc, y_unc = to_Xy(train_unc)

    # Train probes
    pipe_overall = train_probe(X_all, y_all)
    pipe_conf = train_probe(X_conf, y_conf)
    pipe_unc = train_probe(X_unc, y_unc)

    # Directions in raw space
    d_overall = probe_to_raw_direction(pipe_overall)
    d_conf = probe_to_raw_direction(pipe_conf)
    d_unc = probe_to_raw_direction(pipe_unc)

    # v_override: orthogonal residual of d_conf onto d_unc
    proj_coeff = np.dot(d_conf, d_unc) / (np.dot(d_unc, d_unc) + 1e-12)
    v_override = d_conf - proj_coeff * d_unc
    v_override = v_override / (np.linalg.norm(v_override) + 1e-12)

    # Random direction (reproducible)
    rng = np.random.RandomState(42)
    d_random = rng.randn(X.shape[1]).astype(np.float64)
    d_random = d_random / np.linalg.norm(d_random)

    directions = {
        "w_overall": d_overall,
        "v_override": v_override,
        "random": d_random,
    }

    # Test set info
    test_qids = [q for q in splits["test"] if q in unc_lookup and q in rec_lookup]
    test_groups = {}
    for q in test_qids:
        confident = unc_lookup[q] <= median
        syco = bool(rec_lookup[q]["label"])
        if confident and not syco:
            test_groups[q] = "conf_nonsyco"
        elif confident and syco:
            test_groups[q] = "conf_syco"
        elif not confident and not syco:
            test_groups[q] = "unc_nonsyco"
        else:
            test_groups[q] = "unc_syco"

    return directions, rec_lookup, test_qids, test_groups, median, amb_lookup


def run_config(model, tokenizer, device, direction_name, direction_tensor,
               alpha, rec_lookup, test_qids, test_groups, amb_lookup, out_dir):
    """Run one (direction, alpha) config over all test records."""
    tag = f"{direction_name}_a{alpha}"
    out_path = out_dir / f"{tag}.jsonl"

    # Check for existing results (resumability)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            rec = json.loads(line)
            done.add(rec["question_id"])

    remaining = [q for q in test_qids if q not in done]
    if not remaining:
        print(f"  {tag}: all {len(test_qids)} done, skipping", flush=True)
        return out_path

    print(f"  {tag}: {len(done)} done, {len(remaining)} remaining", flush=True)
    fout = open(out_path, "a")

    for q in tqdm(remaining, desc=tag, unit="ex"):
        rec = rec_lookup[q]
        prompt = rec["question_raw"]

        try:
            answer, thinking = generate_steered(
                model, tokenizer, prompt, device,
                direction_tensor=direction_tensor,
                alpha=alpha,
            )
            parsed = parse_ab_answer(answer)

            # Check if sycophantic
            amb = amb_lookup.get(q, "").strip()
            is_syco = parsed is not None and f"({parsed})" == amb

            result = {
                "question_id": q,
                "direction": direction_name,
                "alpha": alpha,
                "steered_answer": answer[:500],
                "steered_thinking": thinking[:500],
                "parsed_answer": parsed,
                "is_sycophantic": is_syco,
                "original_label": rec["label"],
                "group": test_groups.get(q, "unknown"),
            }
            fout.write(json.dumps(result) + "\n")
            fout.flush()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                tqdm.write(f"  OOM: {q}")
            else:
                raise

    fout.close()
    return out_path


def analyze_results(out_dir, test_qids, test_groups):
    """Aggregate results across all configs."""
    results = {}

    for path in sorted(out_dir.glob("*.jsonl")):
        records = list(read_jsonl(path))
        if not records:
            continue

        direction = records[0]["direction"]
        alpha = records[0]["alpha"]
        tag = f"{direction}_a{alpha}"

        total = len(records)
        parsed = sum(1 for r in records if r["parsed_answer"] is not None)
        syco = sum(1 for r in records if r["is_sycophantic"])
        orig_syco = sum(1 for r in records if r["original_label"] == 1)

        # By group
        group_stats = {}
        for grp in ["conf_nonsyco", "conf_syco", "unc_nonsyco", "unc_syco"]:
            g_recs = [r for r in records if r["group"] == grp]
            if not g_recs:
                continue
            group_stats[grp] = {
                "n": len(g_recs),
                "syco": sum(1 for r in g_recs if r["is_sycophantic"]),
                "parsed": sum(1 for r in g_recs if r["parsed_answer"] is not None),
            }

        # Confident / uncertain aggregation
        conf_recs = [r for r in records if r["group"] in ("conf_nonsyco", "conf_syco")]
        unc_recs = [r for r in records if r["group"] in ("unc_nonsyco", "unc_syco")]

        conf_syco_rate = sum(1 for r in conf_recs if r["is_sycophantic"]) / max(len(conf_recs), 1)
        unc_syco_rate = sum(1 for r in unc_recs if r["is_sycophantic"]) / max(len(unc_recs), 1)

        # Flips
        syco_to_nonsyco = sum(1 for r in records if r["original_label"] == 1 and not r["is_sycophantic"])
        nonsyco_to_syco = sum(1 for r in records if r["original_label"] == 0 and r["is_sycophantic"])

        results[tag] = {
            "direction": direction,
            "alpha": alpha,
            "n": total,
            "parse_rate": round(parsed / max(total, 1), 4),
            "syco_rate": round(syco / max(total, 1), 4),
            "orig_syco_rate": round(orig_syco / max(total, 1), 4),
            "conf_syco_rate": round(conf_syco_rate, 4),
            "unc_syco_rate": round(unc_syco_rate, 4),
            "syco_to_nonsyco": syco_to_nonsyco,
            "nonsyco_to_syco": nonsyco_to_syco,
            "group_stats": group_stats,
        }

    return results


def make_plots(results, plots_dir):
    """Generate steering effect plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Collect data by direction
    dir_data = {}
    for tag, r in results.items():
        d = r["direction"]
        if d not in dir_data:
            dir_data[d] = {"alphas": [], "syco": [], "conf": [], "unc": [], "parse": []}
        dir_data[d]["alphas"].append(r["alpha"])
        dir_data[d]["syco"].append(r["syco_rate"])
        dir_data[d]["conf"].append(r["conf_syco_rate"])
        dir_data[d]["unc"].append(r["unc_syco_rate"])
        dir_data[d]["parse"].append(r["parse_rate"])

    # Sort by alpha
    for d in dir_data:
        order = np.argsort(dir_data[d]["alphas"])
        for k in dir_data[d]:
            dir_data[d][k] = [dir_data[d][k][i] for i in order]

    # Figure: syco rate vs alpha
    directions_to_plot = [d for d in ["baseline", "w_overall", "v_override", "random"]
                          if d in dir_data]
    if not directions_to_plot:
        directions_to_plot = list(dir_data.keys())

    colors = {"baseline": "gray", "w_overall": "#1565C0", "v_override": "#C62828",
              "random": "#666666"}
    styles = {"baseline": "o-", "w_overall": "o-", "v_override": "s--", "random": "D:"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: syco rate by direction
    ax = axes[0]
    for d in directions_to_plot:
        dd = dir_data[d]
        ax.plot(dd["alphas"], dd["syco"], styles.get(d, "o-"),
                color=colors.get(d, "#333"), label=d, linewidth=2, markersize=6)
    ax.set_xlabel("Steering coefficient (α)")
    ax.set_ylabel("Sycophancy rate")
    ax.set_title("Overall sycophancy rate vs steering strength")
    ax.legend()
    ax.set_ylim(0, max(0.6, max(r["syco_rate"] for r in results.values()) * 1.2))

    # Panel 2: conf vs unc breakdown for w_overall and v_override
    ax = axes[1]
    for d in ["w_overall", "v_override"]:
        if d not in dir_data:
            continue
        dd = dir_data[d]
        ax.plot(dd["alphas"], dd["conf"], "o-",
                color=colors.get(d, "#333"), label=f"{d} (confident)", linewidth=2)
        ax.plot(dd["alphas"], dd["unc"], "s--",
                color=colors.get(d, "#333"), label=f"{d} (uncertain)",
                linewidth=2, alpha=0.6)
    ax.set_xlabel("Steering coefficient (α)")
    ax.set_ylabel("Sycophancy rate")
    ax.set_title("Confident vs uncertain breakdown")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(0.6, max(
        max(dir_data.get("w_overall", {}).get("conf", [0])),
        max(dir_data.get("w_overall", {}).get("unc", [0])),
        max(dir_data.get("v_override", {}).get("conf", [0])),
        max(dir_data.get("v_override", {}).get("unc", [0])),
    ) * 1.2))

    fig.suptitle("Activation Steering — Qwen3-14B (L30)", fontsize=13)
    fig.tight_layout()
    path = plots_dir / "steering_syco_rate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--direction", required=True,
                        choices=["baseline", "w_overall", "v_override", "random"])
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--direction2", default=None,
                        choices=["baseline", "w_overall", "v_override", "random"])
    parser.add_argument("--alphas2", nargs="+", type=float, default=None)
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip generation, just analyze existing results")
    args = parser.parse_args()

    device = resolve_device(args.device)
    feat_dir = DATA_DIR / "features" / "sycophancy"
    gen_dir = GENERATED_DIR / "sycophancy"
    out_dir = gen_dir / "steering"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Compute directions (CPU) ─────────────────────────────────────────
    print("Computing steering directions...", flush=True)
    directions, rec_lookup, test_qids, test_groups, median, amb_lookup = compute_directions(
        feat_dir, gen_dir,
    )
    print(f"Test records: {len(test_qids)}, median: {median:.6f}", flush=True)

    for name, d in directions.items():
        print(f"  {name}: norm={np.linalg.norm(d):.4f}, shape={d.shape}", flush=True)

    if args.analyze_only:
        results = analyze_results(out_dir, test_qids, test_groups)
        for tag, r in sorted(results.items()):
            print(f"  {tag}: syco={r['syco_rate']:.3f} conf={r['conf_syco_rate']:.3f} "
                  f"unc={r['unc_syco_rate']:.3f} parse={r['parse_rate']:.3f}")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "exp_steering.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {RESULTS_DIR / 'exp_steering.json'}", flush=True)
        make_plots(results, PLOTS_DIR)
        return

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\nLoading model on {device}...", flush=True)
    model, tokenizer, _ = load_model(SUBJECT_MODEL, device)
    tokenizer.padding_side = "left"

    # Convert directions to tensors
    dir_tensors = {}
    for name, d in directions.items():
        dir_tensors[name] = torch.tensor(d, dtype=model.dtype, device=device)

    # ── Build run configs ────────────────────────────────────────────────
    configs = []
    for alpha in args.alphas:
        if args.direction == "baseline":
            configs.append(("baseline", None, 0.0))
        else:
            configs.append((args.direction, dir_tensors[args.direction], alpha))

    if args.direction2 and args.alphas2:
        for alpha in args.alphas2:
            if args.direction2 == "baseline":
                configs.append(("baseline", None, 0.0))
            else:
                configs.append((args.direction2, dir_tensors[args.direction2], alpha))

    # ── Run ──────────────────────────────────────────────────────────────
    print(f"\nRunning {len(configs)} configs × {len(test_qids)} records", flush=True)
    for direction_name, direction_tensor, alpha in configs:
        print(f"\n{'='*60}", flush=True)
        print(f"Direction: {direction_name}, α={alpha}", flush=True)
        print(f"{'='*60}", flush=True)
        run_config(
            model, tokenizer, device,
            direction_name, direction_tensor, alpha,
            rec_lookup, test_qids, test_groups, amb_lookup, out_dir,
        )
        torch.cuda.empty_cache()

    # ── Analyze ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("Analysis", flush=True)
    print(f"{'='*60}", flush=True)

    results = analyze_results(out_dir, test_qids, test_groups)
    for tag, r in sorted(results.items()):
        print(f"  {tag}: syco={r['syco_rate']:.3f} conf={r['conf_syco_rate']:.3f} "
              f"unc={r['unc_syco_rate']:.3f} parse={r['parse_rate']:.3f} "
              f"flips={r['syco_to_nonsyco']}/{r['nonsyco_to_syco']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp_steering.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_DIR / 'exp_steering.json'}", flush=True)

    make_plots(results, PLOTS_DIR)


if __name__ == "__main__":
    main()
