"""Extract residual stream activations from Qwen3-30B-A3B intervention responses.

Same-family MoE control: same Qwen3 training pipeline, MoE architecture.
Uses the same chat template and <think> markers as Qwen3-14B.

Usage:
    python scripts/extract_qwen_moe.py --device cuda:0
    python scripts/extract_qwen_moe.py --device cuda:0 --shard-id 0 --num-shards 4
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ACTIVATIONS_DIR,
    QWEN_MOE_LAYER_FRACS,
    QWEN_MOE_MODEL,
    QWEN_MOE_NUM_LAYERS,
)
from src.lib import (
    extract_layer_activations,
    find_thinking_span,
    load_activations,
    load_model,
    read_jsonl,
    resolve_device,
    resolve_layer_indices,
    save_activations,
)

MAX_THINKING_TOKENS = 2500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    layer_indices = resolve_layer_indices(QWEN_MOE_NUM_LAYERS, QWEN_MOE_LAYER_FRACS)
    print(f"Layers: {layer_indices}", flush=True)

    records = list(read_jsonl(
        Path(__file__).resolve().parent.parent / "data" / "generated" / "sycophancy_qwen_moe" / "labeled.jsonl"
    ))
    print(f"Records: {len(records)}", flush=True)

    if args.num_shards > 1:
        records = records[args.shard_id::args.num_shards]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(records)}", flush=True)

    model, tokenizer, _ = load_model(QWEN_MOE_MODEL, device)

    cache_dir = ACTIVATIONS_DIR / "sycophancy_qwen_moe"
    cache_dir.mkdir(parents=True, exist_ok=True)

    extracted = cached = skipped = 0
    for rec in tqdm(records, desc="extract", unit="ex"):
        qid = rec["question_id"]
        cache_path = cache_dir / f"{qid}.npz"

        if not args.no_cache:
            existing = load_activations(cache_path, layer_indices)
            if existing is not None:
                cached += 1
                continue

        thinking = rec.get("thinking", "")
        prompt = rec.get("question_raw", "")
        if not thinking or not thinking.strip() or not prompt:
            skipped += 1
            continue

        try:
            full_ids, start_idx, end_idx = find_thinking_span(
                tokenizer, prompt, thinking,
            )
            end_idx = min(end_idx, start_idx + MAX_THINKING_TOKENS)

            acts = extract_layer_activations(
                model, full_ids, layer_indices,
                start_idx=start_idx, end_idx=end_idx,
            )
            save_activations(cache_path, acts)
            extracted += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                tqdm.write(f"  OOM: {qid}")
                skipped += 1
            else:
                raise

    print(f"\nExtracted: {extracted}, cached: {cached}, skipped: {skipped}", flush=True)


if __name__ == "__main__":
    main()
