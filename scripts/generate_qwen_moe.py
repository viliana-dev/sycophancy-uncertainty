"""Generate control + intervention responses with Qwen3-30B-A3B.

Same-family MoE control for Qwen3-14B. Uses same chat template and
<think> reasoning format. Standalone script (bypasses src/label).

Usage (single GPU):
    python scripts/generate_qwen_moe.py --device cuda

Usage (2 GPUs in parallel):
    for i in 0 1; do
        python scripts/generate_qwen_moe.py --device cuda:$i --shard-id $i --num-shards 2 &
    done
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GENERATED_DIR, QWEN_MOE_MODEL
from src.datasets import load_dataset_for_behavior
from src.lib import (
    apply_chat_template_messages,
    load_model,
    parse_thinking_answer,
    resolve_device,
)
from src.prompts import parse_ab_answer


def generate_batch(model, tokenizer, prompts: list[str], device) -> list[tuple[str, str, str]]:
    """Batched greedy generation. Returns list of (answer, thinking, parsed)."""
    tokenizer.padding_side = "left"
    texts = [apply_chat_template_messages(tokenizer, [{"role": "user", "content": p}]) for p in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    results = []
    for i in range(len(prompts)):
        new_tokens = out_ids[i][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
        thinking, answer = parse_thinking_answer(raw)
        parsed = parse_ab_answer(answer)
        results.append((answer, thinking, parsed))
    return results


def run_pass(
    model, tokenizer, device,
    questions, out_path: Path, use_control: bool, label: str,
    batch_size: int = 1,
):
    """Run one pass (control or intervention), resuming from existing file."""
    done = {}
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                done[rec["question_id"]] = rec
    remaining = [q for q in questions if q.question_id not in done]
    if not remaining:
        print(f"  {label}: all {len(questions)} cached", flush=True)
        return done
    print(f"  {label}: {len(done)} cached, {len(remaining)} to go (batch={batch_size})", flush=True)

    fh = open(out_path, "a")
    n_batches = (len(remaining) + batch_size - 1) // batch_size
    for bi in tqdm(range(n_batches), desc=label, unit="batch"):
        batch_qs = remaining[bi * batch_size : (bi + 1) * batch_size]
        prompts = [q.question_control if use_control else q.question_raw for q in batch_qs]
        try:
            results = generate_batch(model, tokenizer, prompts, device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                tqdm.write(f"  OOM on batch {bi}, falling back to single")
                for q in batch_qs:
                    prompt = q.question_control if use_control else q.question_raw
                    try:
                        results = generate_batch(model, tokenizer, [prompt], device)
                        answer, thinking, parsed = results[0]
                    except RuntimeError:
                        torch.cuda.empty_cache()
                        tqdm.write(f"  OOM: {q.question_id}")
                        continue
                    rec = {
                        "question_id": q.question_id, "source": q.source,
                        "answer": answer, "thinking": thinking, "parsed_answer": parsed,
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    done[q.question_id] = rec
                continue
            raise
        for q, (answer, thinking, parsed) in zip(batch_qs, results):
            rec = {
                "question_id": q.question_id, "source": q.source,
                "answer": answer, "thinking": thinking, "parsed_answer": parsed,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done[q.question_id] = rec
    fh.close()
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    device = resolve_device(args.device)

    questions = load_dataset_for_behavior("sycophancy", n_samples=args.n_samples, seed=args.seed)
    if args.num_shards > 1:
        questions = questions[args.shard_id::args.num_shards]
    print(f"Questions: {len(questions)} (shard {args.shard_id}/{args.num_shards})", flush=True)

    model, tokenizer, n_layers = load_model(QWEN_MOE_MODEL, device)

    out_dir = GENERATED_DIR / "sycophancy_qwen_moe"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""

    print(f"\n=== Control pass ===", flush=True)
    run_pass(model, tokenizer, device, questions,
             out_dir / f"control{suffix}.jsonl", use_control=True, label="control",
             batch_size=args.batch_size)

    print(f"\n=== Intervention pass ===", flush=True)
    run_pass(model, tokenizer, device, questions,
             out_dir / f"intervention{suffix}.jsonl", use_control=False, label="intervention",
             batch_size=args.batch_size)

    print(f"\nDone. Output: {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
