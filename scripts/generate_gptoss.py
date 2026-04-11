"""Generate control + intervention responses with gpt-oss-20b.

Standalone script that bypasses src/generate.py (which depends on
src/label that's not in the repo). Does everything in one pass:
  1. Load questions from Anthropic evals
  2. Load gpt-oss-20b on specified GPU
  3. Control pass (persona stripped)
  4. Intervention pass (with persona)

Labeling + splits are done separately via scripts/prepare_labeled_gptoss.py.

Usage (4 GPUs in parallel):
    for i in 0 1 2 3; do
        python scripts/generate_gptoss.py --device cuda:$i --shard-id $i --num-shards 4 &
    done
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GENERATED_DIR, GPTOSS_MODEL
from src.datasets import load_dataset_for_behavior
from src.lib import (
    apply_chat_template_messages_harmony,
    load_model_gptoss,
    parse_thinking_answer_harmony,
    resolve_device,
)
from src.prompts import parse_ab_answer


def generate_one(model, tokenizer, prompt: str, device) -> tuple[str, str, str]:
    """Single greedy generation. Returns (answer_text, thinking_text, parsed_letter)."""
    messages = [{"role": "user", "content": prompt}]
    text = apply_chat_template_messages_harmony(tokenizer, messages)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
    thinking, answer = parse_thinking_answer_harmony(raw)
    parsed = parse_ab_answer(answer)
    return answer, thinking, parsed


def run_pass(
    model, tokenizer, device,
    questions, out_path: Path, use_control: bool, label: str,
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
    print(f"  {label}: {len(done)} cached, {len(remaining)} to go", flush=True)

    fh = open(out_path, "a")
    for q in tqdm(remaining, desc=label, unit="ex"):
        prompt = q.question_control if use_control else q.question_raw
        try:
            answer, thinking, parsed = generate_one(model, tokenizer, prompt, device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                tqdm.write(f"  OOM: {q.question_id}")
                continue
            raise
        rec = {
            "question_id": q.question_id,
            "source": q.source,
            "answer": answer,
            "thinking": thinking,
            "parsed_answer": parsed,
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
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    device = resolve_device(args.device)

    questions = load_dataset_for_behavior("sycophancy", n_samples=args.n_samples, seed=args.seed)
    if args.num_shards > 1:
        questions = questions[args.shard_id::args.num_shards]
    print(f"Questions: {len(questions)} (shard {args.shard_id}/{args.num_shards})", flush=True)

    model, tokenizer, n_layers = load_model_gptoss(GPTOSS_MODEL, device)

    out_dir = GENERATED_DIR / "sycophancy_gptoss"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""

    print(f"\n=== Control pass ===", flush=True)
    run_pass(model, tokenizer, device, questions,
             out_dir / f"control{suffix}.jsonl", use_control=True, label="control")

    print(f"\n=== Intervention pass ===", flush=True)
    run_pass(model, tokenizer, device, questions,
             out_dir / f"intervention{suffix}.jsonl", use_control=False, label="intervention")

    print(f"\nDone. Output: {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
