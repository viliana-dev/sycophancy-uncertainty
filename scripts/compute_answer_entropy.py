"""Compute model answer entropy as a principled uncertainty metric.

For each control record, build:
    chat_template(question_control) + <think>thinking</think>\n\nThe answer is (

then run a single forward pass and read the next-token logits restricted to
the option tokens (A/B for sycophancy). Entropy of the resulting distribution
is the model's "how confident am I in picking an answer" signal.

This replaces the hand-crafted lexical heuristic in score_uncertainty().

Output: data/generated/sycophancy/answer_entropy.jsonl
        each line: {"question_id": ..., "p_a": ..., "p_b": ..., "entropy": ..., "answer_matching_behavior": ...}

Usage:
    python scripts/compute_answer_entropy.py --device cuda
    python scripts/compute_answer_entropy.py --device cuda --shard-id 0 --num-shards 4
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GENERATED_DIR, GPTOSS_MODEL, SUBJECT_MODEL
from src.datasets import load_dataset_for_behavior
from src.lib import (
    apply_chat_template,
    apply_chat_template_harmony,
    load_model,
    load_model_gptoss,
    read_jsonl,
    resolve_device,
    write_jsonl,
)


ANSWER_PREFIX = "\n\nThe answer is ("
OPTION_LETTERS = ["A", "B"]

# Harmony format: analysis channel closed, final channel opened
HARMONY_ANALYSIS_OPEN = "<|channel|>analysis<|message|>"
HARMONY_END = "<|end|>"
HARMONY_ASSISTANT_START = "<|start|>assistant"
HARMONY_FINAL_OPEN = "<|channel|>final<|message|>"


def get_option_token_ids(tokenizer) -> dict[str, int]:
    """Find single-token IDs for option letters in the model's vocab.

    The token must be the exact letter that appears right after '(' in
    'The answer is (X' — usually a single BPE token.
    """
    out: dict[str, int] = {}
    for letter in OPTION_LETTERS:
        # Encode the full prefix and the prefix+letter; the diff gives us
        # the token id(s) for the letter in this exact context.
        ids_no = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
        ids_yes = tokenizer.encode(ANSWER_PREFIX + letter, add_special_tokens=False)
        if len(ids_yes) != len(ids_no) + 1:
            raise RuntimeError(
                f"Letter {letter!r} did not encode as a single appended token "
                f"(prefix={len(ids_no)}, with letter={len(ids_yes)})"
            )
        out[letter] = ids_yes[-1]
    return out


def build_input_ids(tokenizer, prompt: str, thinking: str, harmony: bool = False) -> torch.Tensor:
    if harmony:
        prefix = apply_chat_template_harmony(tokenizer, prompt)
        full = (
            prefix + HARMONY_ANALYSIS_OPEN + thinking + HARMONY_END
            + HARMONY_ASSISTANT_START + HARMONY_FINAL_OPEN
            + "The answer is ("
        )
    else:
        prefix = apply_chat_template(tokenizer, prompt)
        full = prefix + "<think>" + thinking + "</think>" + ANSWER_PREFIX
    return tokenizer(full, return_tensors="pt")["input_ids"]


@torch.no_grad()
def compute_entropy_one(
    model, tokenizer, prompt: str, thinking: str, option_ids: dict[str, int],
    harmony: bool = False,
) -> dict[str, float] | None:
    if not thinking or not thinking.strip():
        return None
    try:
        ids = build_input_ids(tokenizer, prompt, thinking, harmony=harmony).to(model.device)
        out = model(ids)
        # Logits for the next token after the full prefix.
        last_logits = out.logits[0, -1, :].float()
        ids_a = option_ids["A"]
        ids_b = option_ids["B"]
        pair = torch.tensor([last_logits[ids_a], last_logits[ids_b]])
        probs = torch.softmax(pair, dim=0)
        p_a = float(probs[0])
        p_b = float(probs[1])
        eps = 1e-12
        entropy = -(p_a * math.log(p_a + eps) + p_b * math.log(p_b + eps))
        return {"p_a": p_a, "p_b": p_b, "entropy": entropy}
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--model-family", choices=["qwen", "gptoss"], default="qwen")
    args = parser.parse_args()

    device = resolve_device(args.device)
    is_gptoss = args.model_family == "gptoss"
    dir_suffix = "_gptoss" if is_gptoss else ""

    # Load control responses (we score the model's uncertainty BEFORE persona
    # is introduced — that's the "baseline confidence" we want).
    ctrl_path = GENERATED_DIR / f"sycophancy{dir_suffix}" / "control.jsonl"
    ctrl_records = list(read_jsonl(ctrl_path))
    print(f"Loaded {len(ctrl_records)} control records")

    # Load target questions to get the persona-stripped prompt + amb.
    target_qs = load_dataset_for_behavior("sycophancy", n_samples=5100, seed=42)
    q_lookup = {q.question_id: q for q in target_qs}

    # Shard
    if args.num_shards > 1:
        ctrl_records = ctrl_records[args.shard_id::args.num_shards]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(ctrl_records)} records")

    # Load model
    model_id = args.model or (GPTOSS_MODEL if is_gptoss else SUBJECT_MODEL)
    if is_gptoss:
        model, tokenizer, _ = load_model_gptoss(model_id, device)
    else:
        model, tokenizer, _ = load_model(model_id, device)
    model.eval()

    option_ids = get_option_token_ids(tokenizer)
    print(f"Option token ids: {option_ids}")

    out_dir = GENERATED_DIR / f"sycophancy{dir_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = out_dir / f"answer_entropy{suffix}.jsonl"

    # Resume support: skip qids already processed.
    done = set()
    if out_path.exists():
        for rec in read_jsonl(out_path):
            done.add(rec["question_id"])
        print(f"Resuming: {len(done)} already done")

    skipped = 0
    written = 0
    with open(out_path, "a") as fh:
        for rec in tqdm(ctrl_records, desc="entropy", unit="ex"):
            qid = rec["question_id"]
            if qid in done:
                continue
            q = q_lookup.get(qid)
            if q is None:
                skipped += 1
                continue
            result = compute_entropy_one(
                model, tokenizer, q.question_control, rec.get("thinking", ""),
                option_ids, harmony=is_gptoss,
            )
            if result is None:
                skipped += 1
                continue
            line = {
                "question_id": qid,
                "p_a": result["p_a"],
                "p_b": result["p_b"],
                "entropy": result["entropy"],
                "answer_matching_behavior": q.answer_matching_behavior.strip().strip("()"),
            }
            fh.write(json.dumps(line) + "\n")
            fh.flush()
            written += 1

    print(f"\nWritten: {written}, skipped: {skipped}, output: {out_path}")


if __name__ == "__main__":
    main()
