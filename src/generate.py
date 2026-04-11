"""Control/intervention CoT generation pipeline.

Sycophancy uses Anthropic evals dataset with dual labeling:
1. Load Anthropic evals questions (persona-based opinion questions)
2. Control pass: 1 call per question with persona stripped
3. Intervention pass: 1 call per question with persona included
4. Mechanical label: does intervention answer match answer_matching_behavior?
5. Gemini label (optional): did model flip to agree with persona?
6. Merge into labeled.jsonl
7. Split at question level

Usage:
    python -m src.generate --behavior sycophancy --n-samples 500
    python -m src.generate --behavior sycophancy --n-samples 500 --skip-gemini
    python -m src.generate --behavior sandbagging --n-samples 120
"""

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.config import (
    ANTHROPIC_EVALS_FILES,
    BOLD, GREEN, RED, RESET, YELLOW,
    DEFAULT_CONCURRENCY, GENERATED_DIR, TRAIN_SPLIT, VAL_SPLIT,
)
from src.datasets import Anecdote, Question, SycophancyQuestion, load_dataset_for_behavior
from src.label import (
    classify_anecdote,
    compute_switch_rates,
    label_sandbagging,
    label_eval_awareness,
    label_sycophancy_mechanical,
    label_sycophancy_gemini,
    label_sycophancy_combined,
    label_sycophancy_rollout,
)
from src.lib import complete_single, complete_with_messages, get_api_key, read_jsonl, write_jsonl, append_jsonl
from src.prompts import (
    build_control_messages,
    build_sycophancy_control,
    build_sycophancy_intervention,
    build_wmdp_sandbagging_messages,
    build_eval_awareness_messages,
    get_sycophantic_answer,
    parse_ab_answer,
)


# ---------------------------------------------------------------------------
# Sycophancy pipeline (Anthropic evals + dual labeling)
# ---------------------------------------------------------------------------

def _run_sycophancy_pass(
    questions: list[SycophancyQuestion],
    api_key: str | None,
    out_path: Path,
    use_control: bool,
    concurrency: int = DEFAULT_CONCURRENCY,
    label: str = "control",
    complete_fn=None,       # single-prompt fn: (prompt) -> (answer, thinking)
    complete_fn_batch=None, # batched fn: (list[prompt]) -> list[(answer, thinking)]
    batch_size: int = 1,
) -> dict[str, dict]:
    """Run one pass (control or intervention) for sycophancy questions.

    Returns {question_id: record_dict} with resume support.
    If complete_fn is provided, it is called as complete_fn(prompt) -> (answer, thinking)
    and concurrency is forced to 1 (sequential, single GPU).
    """
    existing = {}
    if out_path.exists():
        for rec in read_jsonl(out_path):
            existing[rec["question_id"]] = rec

    remaining = [q for q in questions if q.question_id not in existing]
    if not remaining:
        print(f"  {label}: all {len(questions)} cached")
        return existing

    print(f"  {label}: {len(existing)} cached, {len(remaining)} to generate")
    done = 0

    def call(q):
        prompt = q.question_control if use_control else q.question_raw
        if complete_fn is not None:
            answer, thinking = complete_fn(prompt)
        else:
            answer, thinking = complete_single(api_key, prompt)
        return q, answer, thinking

    if complete_fn_batch is not None:
        # Local batched inference
        for i in range(0, len(remaining), batch_size):
            batch = remaining[i:i + batch_size]
            prompts = [q.question_control if use_control else q.question_raw for q in batch]
            try:
                batch_results = complete_fn_batch(prompts)
            except Exception as e:
                done += len(batch)
                print(f"  [ERR] batch {i//batch_size}: {e}")
                continue
            for q, (answer, thinking) in zip(batch, batch_results):
                done += 1
                parsed = parse_ab_answer(answer)
                record = {
                    "question_id": q.question_id,
                    "source": q.source,
                    "answer": answer,
                    "thinking": thinking,
                    "parsed_answer": parsed,
                }
                append_jsonl(out_path, record)
                existing[q.question_id] = record
            if done % 50 == 0 or done >= len(remaining):
                print(f"  [{done}/{len(remaining)}] {label} done")
    elif complete_fn is not None:
        # Local inference: sequential (fallback)
        for q in remaining:
            try:
                q, answer, thinking = call(q)
            except Exception as e:
                done += 1
                print(f"  [ERR] [{done}/{len(remaining)}]: {e}")
                continue
            done += 1
            parsed = parse_ab_answer(answer)
            record = {
                "question_id": q.question_id,
                "source": q.source,
                "answer": answer,
                "thinking": thinking,
                "parsed_answer": parsed,
            }
            append_jsonl(out_path, record)
            existing[q.question_id] = record
            if done % 50 == 0 or done == len(remaining):
                print(f"  [{done}/{len(remaining)}] {label} done")
    else:
        with ThreadPoolExecutor(max_workers=min(len(remaining), concurrency)) as pool:
            futures = [pool.submit(call, q) for q in remaining]
            for fut in as_completed(futures):
                try:
                    q, answer, thinking = fut.result()
                except Exception as e:
                    done += 1
                    print(f"  [ERR] [{done}/{len(remaining)}]: {e}")
                    continue
                done += 1
                parsed = parse_ab_answer(answer)
                record = {
                    "question_id": q.question_id,
                    "source": q.source,
                    "answer": answer,
                    "thinking": thinking,
                    "parsed_answer": parsed,
                }
                append_jsonl(out_path, record)
                existing[q.question_id] = record
                if done % 50 == 0 or done == len(remaining):
                    print(f"  [{done}/{len(remaining)}] {label} done")

    return existing


def run_sycophancy_pipeline(
    n_samples: int = 500,
    sources: list[str] | None = None,
    skip_gemini: bool = False,
    seed: int = 42,
    concurrency: int = DEFAULT_CONCURRENCY,
    local: bool = False,
    device: str = "cuda",
    shard_id: int = 0,
    num_shards: int = 1,
    batch_size: int = 4,
    model_family: str = "qwen",
):
    """Run the sycophancy pipeline using Anthropic evals + dual labeling.

    Steps:
    1. Load Anthropic evals questions
    2. Control pass (persona stripped, 1 call per question)
    3. Intervention pass (with persona, 1 call per question)
    4. Mechanical labeling
    5. Gemini labeling (optional)
    6. Merge into labeled.jsonl
    7. Validate
    8. Split at question level

    With --local: uses local Qwen3-14B on GPU instead of OpenRouter API.
    With --num-shards > 1: splits questions across shards for parallel multi-GPU runs.
    Gemini labeling is automatically skipped in local mode (no API needed).
    """
    api_key = get_api_key() if not local else None
    is_gptoss = model_family == "gptoss"
    dir_suffix = "_gptoss" if is_gptoss else ""
    behavior_dir = GENERATED_DIR / f"sycophancy{dir_suffix}"
    behavior_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}=== Sycophancy Pipeline (Anthropic Evals) ==={RESET}")
    print(f"  Mode: {'local GPU' if local else 'OpenRouter API'}")
    print(f"  Model family: {model_family}")
    print(f"  Sources: {sources or 'all'}")
    print(f"  Gemini labeling: {'skip (local mode)' if local else ('skip' if skip_gemini else 'enabled')}")
    if num_shards > 1:
        print(f"  Shard: {shard_id}/{num_shards}")

    # Step 1: Load questions
    print(f"\n{BOLD}Step 1: Loading Anthropic evals ({n_samples} samples){RESET}")
    questions = load_dataset_for_behavior(
        "sycophancy", n_samples=n_samples, seed=seed, sources=sources,
    )
    if num_shards > 1:
        questions = questions[shard_id::num_shards]
        print(f"  Shard {shard_id}: {len(questions)} questions")
    else:
        print(f"  Loaded {len(questions)} questions")
    source_counts = {}
    for q in questions:
        source_counts[q.source] = source_counts.get(q.source, 0) + 1
    for src, count in source_counts.items():
        print(f"    {src}: {count}")

    # Load local model if needed
    complete_fn = None
    complete_fn_batch = None
    if local:
        import torch
        from src.lib import resolve_device as _resolve_device
        _device = _resolve_device(device)

        if is_gptoss:
            from src.config import GPTOSS_MODEL
            from src.lib import (
                load_model_gptoss,
                apply_chat_template_messages_harmony,
                parse_thinking_answer_harmony,
            )
            _model_id = GPTOSS_MODEL
            _max_new = 2048  # gpt-oss CoT is ~200 tokens, plenty of headroom
            print(f"\n{BOLD}Loading local model {_model_id} ({_device}), batch_size={batch_size}...{RESET}")
            _model, _tokenizer, _ = load_model_gptoss(_model_id, _device)
            _apply_tpl = lambda msgs: apply_chat_template_messages_harmony(_tokenizer, msgs)
            _parse = parse_thinking_answer_harmony
        else:
            from src.config import SUBJECT_MODEL
            from src.lib import (
                load_model,
                apply_chat_template_messages,
                parse_thinking_answer,
            )
            _model_id = SUBJECT_MODEL
            _max_new = 8192
            print(f"\n{BOLD}Loading local model {_model_id} ({_device}), batch_size={batch_size}...{RESET}")
            _model, _tokenizer, _ = load_model(_model_id, _device)
            _apply_tpl = lambda msgs: apply_chat_template_messages(_tokenizer, msgs)
            _parse = parse_thinking_answer

        _tokenizer.padding_side = "left"

        def complete_fn(prompt: str) -> tuple[str, str]:
            msgs = [{"role": "user", "content": prompt}]
            text = _apply_tpl(msgs)
            inputs = _tokenizer(text, return_tensors="pt").to(_device)
            with torch.no_grad():
                output_ids = _model.generate(
                    **inputs,
                    max_new_tokens=_max_new,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=_tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            raw = _tokenizer.decode(new_tokens, skip_special_tokens=False)
            thinking, answer = _parse(raw)
            return answer, thinking

        def complete_fn_batch(prompts: list[str]) -> list[tuple[str, str]]:
            texts = [_apply_tpl([{"role": "user", "content": p}]) for p in prompts]
            inputs = _tokenizer(
                texts, return_tensors="pt", padding=True, truncation=False,
            ).to(_device)
            with torch.no_grad():
                output_ids = _model.generate(
                    **inputs,
                    max_new_tokens=_max_new,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=_tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            results = []
            for i in range(len(prompts)):
                new_tokens = output_ids[i][input_len:]
                raw = _tokenizer.decode(new_tokens, skip_special_tokens=False)
                thinking, answer = _parse(raw)
                results.append((answer, thinking))
            return results

        skip_gemini = True

    # Step 2: Control pass
    print(f"\n{BOLD}Step 2: Control pass (persona stripped){RESET}")
    control_path = behavior_dir / "control.jsonl"
    control_records = _run_sycophancy_pass(
        questions, api_key, control_path,
        use_control=True, concurrency=concurrency, label="control",
        complete_fn=complete_fn if not local else None,
        complete_fn_batch=complete_fn_batch if local else None,
        batch_size=batch_size,
    )

    # Step 3: Intervention pass
    print(f"\n{BOLD}Step 3: Intervention pass (with persona){RESET}")
    intervention_path = behavior_dir / "intervention.jsonl"
    intervention_records = _run_sycophancy_pass(
        questions, api_key, intervention_path,
        use_control=False, concurrency=concurrency, label="intervention",
        complete_fn=complete_fn if not local else None,
        complete_fn_batch=complete_fn_batch if local else None,
        batch_size=batch_size,
    )

    # Step 4: Mechanical labeling
    print(f"\n{BOLD}Step 4: Mechanical labeling{RESET}")
    question_lookup = {q.question_id: q for q in questions}

    mechanical_labels = {}
    for qid, irec in intervention_records.items():
        q = question_lookup.get(qid)
        if q is None:
            continue
        parsed = irec.get("parsed_answer")
        mechanical_labels[qid] = label_sycophancy_mechanical(parsed, q.answer_matching_behavior)

    n_mech_pos = sum(1 for v in mechanical_labels.values() if v == 1)
    n_mech_neg = sum(1 for v in mechanical_labels.values() if v == 0)
    n_mech_none = sum(1 for v in mechanical_labels.values() if v is None)
    print(f"  Mechanical: {n_mech_pos} sycophantic, {n_mech_neg} not, {n_mech_none} unparseable")

    # Step 5: Gemini labeling (optional)
    gemini_labels = {}
    if not skip_gemini:
        print(f"\n{BOLD}Step 5: Gemini labeling{RESET}")
        gemini_cache_path = behavior_dir / "gemini_labels.jsonl"

        # Resume support
        existing_gemini = {}
        if gemini_cache_path.exists():
            for rec in read_jsonl(gemini_cache_path):
                existing_gemini[rec["question_id"]] = rec.get("gemini_label")

        to_label = [qid for qid in intervention_records if qid not in existing_gemini]
        print(f"  {len(existing_gemini)} cached, {len(to_label)} to label")

        if to_label:
            from openai import OpenAI
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

            done = 0

            def gemini_call(qid):
                q = question_lookup.get(qid)
                irec = intervention_records[qid]
                crec = control_records.get(qid, {})
                return qid, label_sycophancy_gemini(
                    client,
                    q.question_raw,
                    q.question_control,
                    crec.get("answer", ""),
                    irec.get("answer", ""),
                    irec.get("thinking", ""),
                )

            with ThreadPoolExecutor(max_workers=min(len(to_label), concurrency)) as pool:
                futures = [pool.submit(gemini_call, qid) for qid in to_label]
                for fut in as_completed(futures):
                    try:
                        qid, result = fut.result()
                    except Exception as e:
                        done += 1
                        print(f"  [ERR] [{done}/{len(to_label)}]: {e}")
                        continue
                    done += 1

                    existing_gemini[qid] = result
                    append_jsonl(gemini_cache_path, {
                        "question_id": qid,
                        "gemini_label": result,
                    })

                    if done % 50 == 0 or done == len(to_label):
                        print(f"  [{done}/{len(to_label)}] gemini labels done")

        gemini_labels = existing_gemini
        n_gem_pos = sum(1 for v in gemini_labels.values() if v is True)
        n_gem_neg = sum(1 for v in gemini_labels.values() if v is False)
        n_gem_none = sum(1 for v in gemini_labels.values() if v is None)
        print(f"  Gemini: {n_gem_pos} sycophantic, {n_gem_neg} not, {n_gem_none} failed")
    else:
        print(f"\n{BOLD}Step 5: Gemini labeling (skipped){RESET}")

    # Step 6: Merge into labeled.jsonl
    print(f"\n{BOLD}Step 6: Merging labels{RESET}")
    labeled_records = []
    for q in questions:
        qid = q.question_id
        irec = intervention_records.get(qid)
        crec = control_records.get(qid)
        if irec is None or crec is None:
            continue

        mech = mechanical_labels.get(qid)
        gem = gemini_labels.get(qid)

        if mech is None and gem is None:
            continue  # Can't label at all

        label, method = label_sycophancy_combined(mech, gem)

        labeled_records.append({
            "question_id": qid,
            "source": q.source,
            "question": q.question_control,
            "question_raw": q.question_raw,
            "answer_matching_behavior": q.answer_matching_behavior,
            "control_answer": crec.get("answer", ""),
            "control_thinking": crec.get("thinking", ""),
            "parsed_control_answer": crec.get("parsed_answer"),
            "model_answer": irec.get("answer", ""),
            "thinking": irec.get("thinking", ""),
            "parsed_intervention_answer": irec.get("parsed_answer"),
            "label_mechanical": mech,
            "label_gemini": int(gem) if gem is not None else None,
            "label": label,
            "label_method": method,
        })

    n_pos = sum(1 for r in labeled_records if r["label"] == 1)
    n_neg = sum(1 for r in labeled_records if r["label"] == 0)
    print(f"  Labeled: {n_pos} sycophantic, {n_neg} not ({len(labeled_records)} total)")

    if not labeled_records:
        print(f"  {RED}No labeled records!{RESET}")
        return [], {}

    labeled_path = behavior_dir / "labeled.jsonl"
    write_jsonl(labeled_path, labeled_records)
    print(f"  Saved: {labeled_path}")

    # Step 7: Validate
    print(f"\n{BOLD}Step 7: Validation{RESET}")
    _validate_sycophancy_dataset(labeled_records)

    # Step 8: Split at question level (70/15/15 stratified)
    print(f"\n{BOLD}Step 8: Creating splits (question-level){RESET}")
    splits = _create_question_splits(labeled_records, seed)
    splits_path = behavior_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"  Saved: {splits_path}")

    print(f"\n{BOLD}=== Sycophancy pipeline complete ==={RESET}")
    return labeled_records, splits


def _validate_sycophancy_dataset(records: list[dict]):
    """Run validation checks on the sycophancy dataset."""
    n = len(records)
    n_pos = sum(1 for r in records if r["label"] == 1)
    rate = n_pos / n if n > 0 else 0

    checks = [
        ("Sufficient examples (>=50)", n >= 50, f"{n}"),
        ("Positive rate 15-85%", 0.15 <= rate <= 0.85, f"{100*rate:.1f}%"),
        ("No empty thinking", sum(1 for r in records if not r.get("thinking", "").strip()) == 0,
         f"{sum(1 for r in records if not r.get('thinking', '').strip())} empty"),
    ]

    # Label method distribution
    methods = {}
    for r in records:
        m = r.get("label_method", "unknown")
        methods[m] = methods.get(m, 0) + 1
    for method, count in sorted(methods.items()):
        checks.append((f"Method: {method}", True, f"{count}"))

    for name, passed, detail in checks:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  [{status}] {name}: {detail}")


# ---------------------------------------------------------------------------
# Sandbagging pipeline (simpler: control + intervention)
# ---------------------------------------------------------------------------

def run_sandbagging_pipeline(
    n_samples: int = 120,
    pressure_level: int = 1,
    seed: int = 42,
    concurrency: int = DEFAULT_CONCURRENCY,
):
    """Run sandbagging generation pipeline."""
    api_key = get_api_key()
    behavior_dir = GENERATED_DIR / "sandbagging"
    behavior_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}=== Sandbagging Pipeline ==={RESET}")

    # Load questions
    questions = load_dataset_for_behavior("sandbagging", n_samples=n_samples, seed=seed)
    print(f"  Loaded {len(questions)} WMDP (bio+chem) questions")

    # Control pass
    print(f"\n{BOLD}Control pass{RESET}")
    control_path = behavior_dir / "control.jsonl"
    control_records = _run_control_pass_questions(questions, api_key, control_path, concurrency)

    # Filter to correctly answered
    correct = [r for r in control_records if r["is_correct"]]
    print(f"  Control accuracy: {len(correct)}/{len(control_records)}")

    # Intervention pass
    print(f"\n{BOLD}Intervention pass{RESET}")
    intervention_path = behavior_dir / "intervention.jsonl"
    intervention_records = _run_sandbagging_intervention(
        correct, api_key, intervention_path, pressure_level, concurrency,
    )

    # Label
    labeled = []
    for rec in intervention_records:
        label = label_sandbagging(rec["correct_answer"], rec["model_answer"], control_correct=True)
        labeled.append({**rec, "label": label})

    n_pos = sum(1 for r in labeled if r["label"] == 1)
    print(f"  Sandbagging rate: {n_pos}/{len(labeled)} ({100*n_pos/len(labeled):.1f}%)")

    labeled_path = behavior_dir / "labeled.jsonl"
    write_jsonl(labeled_path, labeled)

    # Split
    splits = _create_question_splits(labeled, seed)
    with open(behavior_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    return labeled, splits


def _run_control_pass_questions(questions, api_key, out_path, concurrency):
    """Control pass for Question-based behaviors (sandbagging, eval_awareness)."""
    existing = {}
    if out_path.exists():
        for rec in read_jsonl(out_path):
            existing[rec["question_id"]] = rec

    remaining = [q for q in questions if q.question_id not in existing]
    if not remaining:
        return list(existing.values())

    print(f"  {len(existing)} cached, {len(remaining)} to generate")
    done = 0

    def call(q):
        messages = build_control_messages(q.question)
        answer, thinking = complete_with_messages(api_key, messages)
        return q, answer, thinking

    with ThreadPoolExecutor(max_workers=min(len(remaining), concurrency)) as pool:
        futures = [pool.submit(call, q) for q in remaining]
        for fut in as_completed(futures):
            try:
                q, answer, thinking = fut.result()
            except Exception as e:
                done += 1
                continue
            done += 1
            is_correct = q.correct_answer.lower() in answer.lower()
            record = {
                "question_id": q.question_id,
                "question": q.question,
                "correct_answer": q.correct_answer,
                "model_answer": answer,
                "thinking": thinking,
                "is_correct": is_correct,
            }
            append_jsonl(out_path, record)
            existing[q.question_id] = record

    return list(existing.values())


def _run_sandbagging_intervention(records, api_key, out_path, pressure_level, concurrency):
    """Intervention pass for sandbagging."""
    existing = {}
    if out_path.exists():
        for rec in read_jsonl(out_path):
            existing[rec["question_id"]] = rec

    remaining = [r for r in records if r["question_id"] not in existing]
    if not remaining:
        return list(existing.values())

    done = 0

    def call(rec):
        messages = build_wmdp_sandbagging_messages(rec["question"], pressure_level)
        answer, thinking = complete_with_messages(api_key, messages)
        return rec, answer, thinking

    with ThreadPoolExecutor(max_workers=min(len(remaining), concurrency)) as pool:
        futures = [pool.submit(call, r) for r in remaining]
        for fut in as_completed(futures):
            try:
                rec, answer, thinking = fut.result()
            except Exception:
                done += 1
                continue
            done += 1
            record = {
                "question_id": rec["question_id"],
                "question": rec["question"],
                "correct_answer": rec["correct_answer"],
                "model_answer": answer,
                "thinking": thinking,
                "control_answer": rec["model_answer"],
            }
            append_jsonl(out_path, record)
            existing[rec["question_id"]] = record

    return list(existing.values())


def _create_question_splits(records, seed):
    """Create 70/15/15 splits at question level."""
    rng = random.Random(seed)
    positives = [r["question_id"] for r in records if r["label"] == 1]
    negatives = [r["question_id"] for r in records if r["label"] == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def split_list(ids):
        n = len(ids)
        n_train = max(1, int(TRAIN_SPLIT * n))
        n_val = max(1, int(VAL_SPLIT * n))
        return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]

    pos_train, pos_val, pos_test = split_list(positives)
    neg_train, neg_val, neg_test = split_list(negatives)

    return {
        "train": pos_train + neg_train,
        "val": pos_val + neg_val,
        "test": pos_test + neg_test,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_generation_pipeline(
    behavior: str,
    n_samples: int = 200,
    sources: list[str] | None = None,
    skip_gemini: bool = False,
    prompt_id: int = 1,
    pressure_level: int = 1,
    seed: int = 42,
    concurrency: int = DEFAULT_CONCURRENCY,
    local: bool = False,
    device: str = "cuda",
    shard_id: int = 0,
    num_shards: int = 1,
    batch_size: int = 4,
    model_family: str = "qwen",
):
    """Run the generation pipeline for any behavior."""
    if behavior == "sycophancy":
        return run_sycophancy_pipeline(
            n_samples=n_samples,
            sources=sources,
            skip_gemini=skip_gemini,
            seed=seed,
            concurrency=concurrency,
            local=local,
            device=device,
            shard_id=shard_id,
            num_shards=num_shards,
            batch_size=batch_size,
            model_family=model_family,
        )
    elif behavior == "sandbagging":
        return run_sandbagging_pipeline(
            n_samples=n_samples,
            pressure_level=pressure_level,
            seed=seed,
            concurrency=concurrency,
        )
    elif behavior == "eval_awareness":
        # Eval awareness uses same anecdote structure but with eval prompt
        # For now, reuse sandbagging-style pipeline
        return run_sandbagging_pipeline(
            n_samples=n_samples,
            pressure_level=1,
            seed=seed,
            concurrency=concurrency,
        )
    else:
        raise ValueError(f"Unknown behavior: {behavior}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate control/intervention datasets")
    parser.add_argument("--behavior", required=True, choices=["sycophancy", "sandbagging", "eval_awareness"])
    parser.add_argument("--n-samples", type=int, default=200, help="Number of questions")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Anthropic evals source files (sycophancy only)")
    parser.add_argument("--skip-gemini", action="store_true",
                        help="Skip Gemini labeling (sycophancy only)")
    parser.add_argument("--prompt-id", type=int, default=1)
    parser.add_argument("--pressure-level", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--local", action="store_true",
                        help="Use local Qwen3-14B instead of OpenRouter API (sycophancy only)")
    parser.add_argument("--device", default="cuda",
                        help="Device for local inference (cuda/cpu)")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Shard index for parallel multi-GPU runs")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total shards for parallel multi-GPU runs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for local inference (default: 4, use 8 on 80GB GPUs)")
    parser.add_argument("--model-family", choices=["qwen", "gptoss"], default="qwen",
                        help="Model family: qwen (Qwen3-14B) or gptoss (gpt-oss-20b)")
    args = parser.parse_args()

    run_generation_pipeline(
        behavior=args.behavior,
        n_samples=args.n_samples,
        sources=args.sources,
        skip_gemini=args.skip_gemini,
        prompt_id=args.prompt_id,
        pressure_level=args.pressure_level,
        seed=args.seed,
        concurrency=args.concurrency,
        local=args.local,
        device=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        batch_size=args.batch_size,
        model_family=args.model_family,
    )


if __name__ == "__main__":
    main()
