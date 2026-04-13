"""MoE routing analysis: do confident and uncertain records use different experts?

Hooks into gpt-oss-20b router at every layer, records top-4 expert assignments
for thinking tokens, then compares expert usage distributions across
(confident/uncertain × syco/nonsyco) groups.

Requires GPU (single forward pass per record, no generation).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/exp_moe_routing.py
"""

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    GENERATED_DIR, GPTOSS_MODEL, GPTOSS_NUM_LAYERS, PLOTS_DIR, RESULTS_DIR,
)
from src.lib import (
    find_analysis_span_harmony,
    load_model_gptoss,
    read_jsonl,
    resolve_device,
)

NUM_EXPERTS = 32
TOP_K = 4
NUM_LAYERS = GPTOSS_NUM_LAYERS  # 24

GROUP_NAMES = {
    0: "confident_nonsyco",
    1: "confident_syco",
    2: "uncertain_nonsyco",
    3: "uncertain_syco",
}


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base e) between two distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / (m + 1e-12) + 1e-12)))
    kl_qm = float(np.sum(q * np.log(q / (m + 1e-12) + 1e-12)))
    return 0.5 * kl_pm + 0.5 * kl_qm


def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-thinking-tokens", type=int, default=2500)
    args = parser.parse_args()

    device = resolve_device(args.device)
    gen_dir = GENERATED_DIR / "sycophancy_gptoss"

    # ── Load data ────────────────────────────────────────────────────────
    records = list(read_jsonl(gen_dir / "labeled.jsonl"))
    rec_lookup = {r["question_id"]: r for r in records}
    with open(gen_dir / "splits.json") as f:
        splits = json.load(f)

    # Entropy
    ent_path = gen_dir / "answer_entropy.jsonl"
    unc_lookup = {}
    if ent_path.exists():
        for rec in read_jsonl(ent_path):
            unc_lookup[rec["question_id"]] = rec["entropy"]
        print(f"Loaded {len(unc_lookup)} entropy records", flush=True)
    else:
        unc_lookup = {r["question_id"]: r["uncertainty_score"] for r in records}
        print("Using heuristic uncertainty", flush=True)

    # Median (trainval only)
    trainval_qids = [q for q in splits["train"] + splits["val"] if q in unc_lookup]
    median = float(np.median([unc_lookup[q] for q in trainval_qids]))
    print(f"Uncertainty median: {median:.6f}", flush=True)

    test_qids = [q for q in splits["test"] if q in unc_lookup and q in rec_lookup]
    print(f"Test records: {len(test_qids)}", flush=True)

    # Assign groups
    test_groups = {}
    for q in test_qids:
        confident = unc_lookup[q] <= median
        syco = bool(rec_lookup[q]["label"])
        if confident and not syco:
            test_groups[q] = 0
        elif confident and syco:
            test_groups[q] = 1
        elif not confident and not syco:
            test_groups[q] = 2
        else:
            test_groups[q] = 3

    for g, name in GROUP_NAMES.items():
        n = sum(1 for v in test_groups.values() if v == g)
        print(f"  {name}: {n}", flush=True)

    # ── Load model ───────────────────────────────────────────────────────
    model, tokenizer, _ = load_model_gptoss(GPTOSS_MODEL, device)
    model.eval()

    # ── Hook routers ─────────────────────────────────────────────────────
    router_data = {}  # layer_idx -> list of (router_indices_cpu,)

    def make_hook(layer_idx):
        def hook(module, input, output):
            router_logits, router_scores, router_indices = output
            router_data[layer_idx] = router_indices.detach().cpu()
        return hook

    hooks = []
    for layer_idx in range(NUM_LAYERS):
        router = model.model.layers[layer_idx].mlp.router
        h = router.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # ── Forward pass per test record ─────────────────────────────────────
    # Per-record: expert_freq[layer] = (32,) count of how often each expert is selected
    record_freqs = {}  # qid -> {layer: np.ndarray(32)}

    skipped = 0
    with torch.no_grad():
        for q in tqdm(test_qids, desc="routing", unit="ex"):
            rec = rec_lookup[q]
            thinking = rec.get("thinking", "")
            prompt = rec.get("question_raw", "")
            if not thinking or not thinking.strip() or not prompt:
                skipped += 1
                continue

            try:
                full_ids, start_idx, end_idx = find_analysis_span_harmony(
                    tokenizer, prompt, thinking,
                )
                end_idx = min(end_idx, start_idx + args.max_thinking_tokens)
                n_thinking = end_idx - start_idx
                if n_thinking < 2:
                    skipped += 1
                    continue

                # Forward pass
                full_ids = full_ids.to(device)
                router_data.clear()
                model(full_ids)

                # Extract expert indices for thinking tokens only
                freqs = {}
                for layer_idx in range(NUM_LAYERS):
                    indices = router_data[layer_idx]  # (seq_len, 4)
                    # indices may be (seq_len, 4) or (1, seq_len, 4)
                    if indices.ndim == 3:
                        indices = indices[0]
                    thinking_indices = indices[start_idx:end_idx, :]  # (n_thinking, 4)
                    freq = np.zeros(NUM_EXPERTS, dtype=np.float32)
                    for expert_id in thinking_indices.numpy().flatten():
                        freq[expert_id] += 1
                    freq /= (freq.sum() + 1e-12)  # normalize to distribution
                    freqs[layer_idx] = freq

                record_freqs[q] = freqs

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    gc.collect()
                    tqdm.write(f"  OOM: {q}")
                    skipped += 1
                else:
                    raise

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"\nProcessed: {len(record_freqs)}, skipped: {skipped}", flush=True)

    # ══════════════════════════════════════════════════════════════════════
    # Analysis
    # ══════════════════════════════════════════════════════════════════════

    # A. Average expert frequency per group per layer
    group_freqs = {}  # group -> layer -> (32,) mean distribution
    for g_idx, name in GROUP_NAMES.items():
        g_qids = [q for q in record_freqs if test_groups.get(q) == g_idx]
        if not g_qids:
            continue
        layer_stacks = {l: [] for l in range(NUM_LAYERS)}
        for q in g_qids:
            for l in range(NUM_LAYERS):
                layer_stacks[l].append(record_freqs[q][l])
        group_freqs[name] = {
            l: np.mean(layer_stacks[l], axis=0) for l in range(NUM_LAYERS)
        }

    # B. JS divergence: confident vs uncertain, syco vs nonsyco
    js_conf_unc = []  # per layer
    js_syco_nonsyco = []
    for l in range(NUM_LAYERS):
        # conf = avg of conf_nonsyco + conf_syco
        p_conf = (group_freqs["confident_nonsyco"][l] + group_freqs["confident_syco"][l]) / 2
        p_unc = (group_freqs["uncertain_nonsyco"][l] + group_freqs["uncertain_syco"][l]) / 2
        js_conf_unc.append(js_divergence(p_conf, p_unc))

        p_nonsyco = (group_freqs["confident_nonsyco"][l] + group_freqs["uncertain_nonsyco"][l]) / 2
        p_syco = (group_freqs["confident_syco"][l] + group_freqs["uncertain_syco"][l]) / 2
        js_syco_nonsyco.append(js_divergence(p_nonsyco, p_syco))

    print(f"\n=== JS Divergence per layer ===")
    print(f"{'Layer':<8} {'Conf vs Unc':<16} {'Syco vs NonSyco':<16} {'Ratio':<8}")
    print("-" * 50)
    for l in range(NUM_LAYERS):
        ratio = js_conf_unc[l] / (js_syco_nonsyco[l] + 1e-12)
        print(f"L{l:<6} {js_conf_unc[l]:.6f}        {js_syco_nonsyco[l]:.6f}        {ratio:.2f}")

    # C. Jaccard overlap of top-8 dominant experts
    top_n = 8
    jaccard_conf_unc = []
    jaccard_syco_nonsyco = []
    for l in range(NUM_LAYERS):
        conf_top = set(np.argsort(
            group_freqs["confident_nonsyco"][l] + group_freqs["confident_syco"][l]
        )[-top_n:])
        unc_top = set(np.argsort(
            group_freqs["uncertain_nonsyco"][l] + group_freqs["uncertain_syco"][l]
        )[-top_n:])
        jaccard_conf_unc.append(jaccard(conf_top, unc_top))

        ns_top = set(np.argsort(
            group_freqs["confident_nonsyco"][l] + group_freqs["uncertain_nonsyco"][l]
        )[-top_n:])
        s_top = set(np.argsort(
            group_freqs["confident_syco"][l] + group_freqs["uncertain_syco"][l]
        )[-top_n:])
        jaccard_syco_nonsyco.append(jaccard(ns_top, s_top))

    print(f"\n=== Jaccard overlap (top-{top_n} experts) ===")
    print(f"{'Layer':<8} {'Conf vs Unc':<16} {'Syco vs NonSyco':<16}")
    print("-" * 42)
    for l in range(NUM_LAYERS):
        print(f"L{l:<6} {jaccard_conf_unc[l]:.3f}           {jaccard_syco_nonsyco[l]:.3f}")

    # D. Per-record routing entropy
    def routing_entropy(freq):
        freq = freq + 1e-12
        return float(-np.sum(freq * np.log(freq)))

    group_entropies = {name: {l: [] for l in range(NUM_LAYERS)} for name in GROUP_NAMES.values()}
    for q in record_freqs:
        g = test_groups.get(q)
        if g is None:
            continue
        name = GROUP_NAMES[g]
        for l in range(NUM_LAYERS):
            group_entropies[name][l].append(routing_entropy(record_freqs[q][l]))

    print(f"\n=== Mean routing entropy per group (averaged over layers) ===")
    for name in GROUP_NAMES.values():
        all_ent = [np.mean(group_entropies[name][l]) for l in range(NUM_LAYERS)]
        print(f"  {name:<24} {np.mean(all_ent):.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════════
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: JS divergence per layer
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    layers = list(range(NUM_LAYERS))
    ax.plot(layers, js_conf_unc, "o-", color="#1565C0", label="Confident vs Uncertain", linewidth=2)
    ax.plot(layers, js_syco_nonsyco, "s--", color="#C62828", label="Syco vs Non-syco", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Jensen-Shannon Divergence")
    ax.set_title("MoE Routing Divergence — gpt-oss-20b")
    ax.legend()
    ax.set_xticks(layers)
    fig.tight_layout()
    div_path = PLOTS_DIR / "moe_routing_divergence.png"
    fig.savefig(div_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {div_path}", flush=True)

    # Figure 2: Jaccard overlap per layer
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(layers, jaccard_conf_unc, "o-", color="#1565C0", label="Confident vs Uncertain", linewidth=2)
    ax.plot(layers, jaccard_syco_nonsyco, "s--", color="#C62828", label="Syco vs Non-syco", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Jaccard Similarity (top-{top_n} experts)")
    ax.set_title("Expert Set Overlap — gpt-oss-20b")
    ax.legend()
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    jac_path = PLOTS_DIR / "moe_routing_jaccard.png"
    fig.savefig(jac_path, dpi=150)
    plt.close(fig)
    print(f"Saved {jac_path}", flush=True)

    # Figure 3: Heatmap of expert usage per group (select layers)
    show_layers = [0, 6, 12, 18, 21, 23]
    fig, axes = plt.subplots(len(show_layers), 4, figsize=(16, len(show_layers) * 2.2))
    for row, l in enumerate(show_layers):
        for col, name in enumerate(GROUP_NAMES.values()):
            ax = axes[row, col]
            freq = group_freqs[name][l]
            ax.bar(range(NUM_EXPERTS), freq, color=["#2196F3", "#F44336", "#64B5F6", "#EF9A9A"][col],
                   edgecolor="none", width=0.8)
            if row == 0:
                ax.set_title(name.replace("_", "\n"), fontsize=9)
            if col == 0:
                ax.set_ylabel(f"L{l}", fontsize=10)
            ax.set_xlim(-0.5, 31.5)
            ax.set_ylim(0, max(0.08, freq.max() * 1.2))
            ax.tick_params(axis="x", labelsize=5)
            ax.tick_params(axis="y", labelsize=6)
            if row < len(show_layers) - 1:
                ax.set_xticklabels([])

    fig.suptitle("Expert Usage Distribution — gpt-oss-20b", fontsize=13, y=1.01)
    fig.tight_layout()
    heat_path = PLOTS_DIR / "moe_routing_heatmap.png"
    fig.savefig(heat_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {heat_path}", flush=True)

    # ── Save JSON ────────────────────────────────────────────────────────
    out = {
        "model": "gpt-oss-20b",
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "num_layers": NUM_LAYERS,
        "n_test_processed": len(record_freqs),
        "n_skipped": skipped,
        "uncertainty_median": median,
        "group_sizes": {
            name: sum(1 for q in record_freqs if test_groups.get(q) == g)
            for g, name in GROUP_NAMES.items()
        },
        "js_divergence_per_layer": {
            "conf_vs_unc": [round(v, 6) for v in js_conf_unc],
            "syco_vs_nonsyco": [round(v, 6) for v in js_syco_nonsyco],
        },
        "jaccard_top8_per_layer": {
            "conf_vs_unc": [round(v, 4) for v in jaccard_conf_unc],
            "syco_vs_nonsyco": [round(v, 4) for v in jaccard_syco_nonsyco],
        },
        "mean_js_conf_vs_unc": round(float(np.mean(js_conf_unc)), 6),
        "mean_js_syco_vs_nonsyco": round(float(np.mean(js_syco_nonsyco)), 6),
        "mean_jaccard_conf_vs_unc": round(float(np.mean(jaccard_conf_unc)), 4),
        "mean_jaccard_syco_vs_nonsyco": round(float(np.mean(jaccard_syco_nonsyco)), 4),
        "mean_routing_entropy": {
            name: round(float(np.mean([np.mean(group_entropies[name][l]) for l in range(NUM_LAYERS)])), 4)
            for name in GROUP_NAMES.values()
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "exp_moe_routing.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
