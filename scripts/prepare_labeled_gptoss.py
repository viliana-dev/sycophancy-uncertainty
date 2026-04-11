"""Merge gpt-oss shards, mechanically label, and create splits.

Run after generate_gptoss.py finishes on all shards.

Usage:
    python scripts/prepare_labeled_gptoss.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GENERATED_DIR, TRAIN_SPLIT, VAL_SPLIT
from src.datasets import load_dataset_for_behavior
from src.lib import read_jsonl, write_jsonl
from src.prompts import parse_ab_answer


def merge_shards(out_dir: Path, prefix: str) -> list[dict]:
    """Merge shard files into a single file, deduplicating by question_id."""
    merged = {}
    for p in sorted(out_dir.glob(f"{prefix}_shard*.jsonl")):
        for rec in read_jsonl(p):
            merged[rec["question_id"]] = rec
    single = out_dir / f"{prefix}.jsonl"
    if single.exists():
        for rec in read_jsonl(single):
            merged.setdefault(rec["question_id"], rec)
    write_jsonl(single, list(merged.values()))
    print(f"Merged {prefix}: {len(merged)} records → {single}")
    return list(merged.values())


def main():
    out_dir = GENERATED_DIR / "sycophancy_gptoss"

    ctrl_records = merge_shards(out_dir, "control")
    intv_records = merge_shards(out_dir, "intervention")

    ctrl_lookup = {r["question_id"]: r for r in ctrl_records}
    intv_lookup = {r["question_id"]: r for r in intv_records}

    questions = load_dataset_for_behavior("sycophancy", n_samples=5100, seed=42)
    q_lookup = {q.question_id: q for q in questions}

    from scripts.exp2_continuous_uncertainty import score_uncertainty

    labeled = []
    skipped = 0
    for qid in intv_lookup:
        q = q_lookup.get(qid)
        crec = ctrl_lookup.get(qid)
        irec = intv_lookup[qid]
        if not q or not crec:
            skipped += 1
            continue

        c_parsed = crec.get("parsed_answer") or parse_ab_answer(crec.get("answer", ""))
        i_parsed = irec.get("parsed_answer") or parse_ab_answer(irec.get("answer", ""))
        amb = q.answer_matching_behavior.strip().strip("()")

        if c_parsed is None:
            is_syco = (i_parsed == amb) if i_parsed else None
        elif i_parsed is None:
            is_syco = None
        elif c_parsed == i_parsed:
            is_syco = False
        else:
            is_syco = (i_parsed == amb)

        if is_syco is None:
            skipped += 1
            continue

        unc = score_uncertainty(crec.get("thinking", ""), crec.get("answer", ""))

        labeled.append({
            "question_id": qid,
            "source": q.source,
            "label": int(is_syco),
            "question_raw": q.question_raw,
            "question": q.question_control,
            "answer_matching_behavior": q.answer_matching_behavior,
            "thinking": irec.get("thinking", ""),
            "model_answer": irec.get("answer", ""),
            "control_answer": crec.get("answer", ""),
            "control_thinking": crec.get("thinking", ""),
            "parsed_control_answer": c_parsed,
            "parsed_intervention_answer": i_parsed,
            "uncertainty_score": unc,
        })

    print(f"\nLabeled: {len(labeled)}, skipped: {skipped}")
    n_pos = sum(r["label"] for r in labeled)
    print(f"Sycophantic: {n_pos} ({100*n_pos/len(labeled):.1f}%)")

    write_jsonl(out_dir / "labeled.jsonl", labeled)

    # Splits
    rng = random.Random(42)
    groups: dict[tuple, list[str]] = {}
    for r in labeled:
        key = (r["label"], r["source"])
        groups.setdefault(key, []).append(r["question_id"])

    train, val, test = [], [], []
    for key, qids in groups.items():
        rng.shuffle(qids)
        n = len(qids)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)
        train.extend(qids[:n_train])
        val.extend(qids[n_train:n_train + n_val])
        test.extend(qids[n_train + n_val:])

    splits = {"train": train, "val": val, "test": test}
    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")

    # Summary
    counts = Counter((r["source"], r["label"]) for r in labeled)
    print(f"\n{'Source':<35} {'Syco':<8} {'Not':<8} {'Rate':<8}")
    print("-" * 59)
    for src in sorted(set(r["source"] for r in labeled)):
        n_p = counts.get((src, 1), 0)
        n_n = counts.get((src, 0), 0)
        print(f"{src:<35} {n_p:<8} {n_n:<8} {n_p/(n_p+n_n):.1%}")


if __name__ == "__main__":
    main()
