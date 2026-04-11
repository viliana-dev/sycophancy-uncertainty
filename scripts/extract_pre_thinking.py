"""Extract "K=0" features: residual stream at the last position before
the model starts reasoning (thinking/analysis channel opened but no content
tokens yet).

For each record, build:
  Qwen:   chat_template(prompt) + <think>
  gptoss: chat_template(prompt) + <|channel|>analysis<|message|>

and run a single forward pass to capture the hidden state at the final
token position for each target layer.

This gives the probe's "pre-reasoning" baseline — the model's activation
state from the intervention prompt alone, before any CoT.

Output: data/features/sycophancy[_gptoss]/L{layer}_K0.npz
        keys: 'X' (n, hidden_dim), 'qids' (n,)

Usage:
    python scripts/extract_pre_thinking.py --device cuda
    python scripts/extract_pre_thinking.py --device cuda --model-family gptoss
    python scripts/extract_pre_thinking.py --device cuda --shard-id 0 --num-shards 4
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ACTIVATIONS_DIR,
    DATA_DIR,
    DEFAULT_LAYER_FRACS,
    GENERATED_DIR,
    GPTOSS_HIDDEN_SIZE,
    GPTOSS_LAYER_FRACS,
    GPTOSS_MODEL,
    GPTOSS_NUM_LAYERS,
    HIDDEN_SIZE,
    NUM_LAYERS,
    SUBJECT_MODEL,
)
from src.lib import (
    extract_layer_activations,
    read_jsonl,
    resolve_device,
    resolve_layer_indices,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-family", choices=["qwen", "gptoss"], default="qwen")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    device = resolve_device(args.device)
    is_gptoss = args.model_family == "gptoss"
    suffix = "_gptoss" if is_gptoss else ""

    if is_gptoss:
        from src.lib import load_model_gptoss, apply_chat_template_harmony
        model_id = GPTOSS_MODEL
        hidden_size = GPTOSS_HIDDEN_SIZE
        layer_fracs = GPTOSS_LAYER_FRACS
        n_layers = GPTOSS_NUM_LAYERS
        model, tokenizer, _ = load_model_gptoss(model_id, device)

        def build_pre_thinking_ids(prompt: str) -> torch.Tensor:
            prefix = apply_chat_template_harmony(tokenizer, prompt)
            text = prefix + "<|channel|>analysis<|message|>"
            return tokenizer(text, return_tensors="pt")["input_ids"]
    else:
        from src.lib import load_model, apply_chat_template
        model_id = SUBJECT_MODEL
        hidden_size = HIDDEN_SIZE
        layer_fracs = DEFAULT_LAYER_FRACS
        n_layers = NUM_LAYERS
        model, tokenizer, _ = load_model(model_id, device)

        def build_pre_thinking_ids(prompt: str) -> torch.Tensor:
            prefix = apply_chat_template(tokenizer, prompt)
            text = prefix + "<think>"
            return tokenizer(text, return_tensors="pt")["input_ids"]

    layer_indices = resolve_layer_indices(n_layers, layer_fracs)
    print(f"Model: {model_id}, layers: {layer_indices}, hidden: {hidden_size}", flush=True)

    labeled_path = GENERATED_DIR / f"sycophancy{suffix}" / "labeled.jsonl"
    records = list(read_jsonl(labeled_path))
    print(f"Records: {len(records)}", flush=True)

    if args.num_shards > 1:
        records = records[args.shard_id::args.num_shards]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(records)}", flush=True)

    out_dir = DATA_DIR / "features" / f"sycophancy{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(records)
    matrices = {layer: np.zeros((n, hidden_size), dtype=np.float32) for layer in layer_indices}
    valid = np.zeros(n, dtype=bool)
    qids = [r["question_id"] for r in records]

    for i, rec in enumerate(tqdm(records, desc="pre-thinking", unit="ex")):
        prompt = rec.get("question_raw") or rec.get("question", "")
        if not prompt:
            continue
        try:
            ids = build_pre_thinking_ids(prompt).to(model.device)
            acts = extract_layer_activations(
                model, ids, layer_indices,
                start_idx=ids.shape[1] - 1,
                end_idx=ids.shape[1],
            )
            for layer in layer_indices:
                matrices[layer][i] = acts[layer][0].astype(np.float32)
            valid[i] = True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                tqdm.write(f"  OOM: {rec['question_id']}")
            else:
                raise

    valid_qids = np.array([q for q, v in zip(qids, valid) if v])
    print(f"Valid: {valid.sum()}/{n}", flush=True)

    for layer in layer_indices:
        mat = matrices[layer][valid]
        out_path = out_dir / f"L{layer}_K0.npz"
        np.savez(out_path, X=mat, qids=valid_qids)
        print(f"  Saved {out_path.name}: {mat.shape}", flush=True)


if __name__ == "__main__":
    main()
