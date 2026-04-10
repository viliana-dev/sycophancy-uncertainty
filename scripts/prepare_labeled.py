"""Prepare labeled.jsonl and splits.json without loading the model.

Run this locally to verify labels and splits before sending to GPU machine.

Usage:
    python scripts/prepare_labeled.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GENERATED_DIR
from src.extract import build_labeled_records, make_splits
from src.lib import write_jsonl


def main():
    records = build_labeled_records()

    out_dir = GENERATED_DIR / "sycophancy"
    labeled_path = out_dir / "labeled.jsonl"
    splits_path = out_dir / "splits.json"

    write_jsonl(labeled_path, records)
    print(f"Saved {len(records)} records to {labeled_path}")

    splits = make_splits(records)
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits to {splits_path}")

    # Summary by source x label
    from collections import Counter
    counts = Counter((r["source"], r["label"]) for r in records)
    print("\n=== Distribution ===")
    print(f"{'Source':<35} {'Syco':<8} {'Not':<8} {'Total':<8} {'Rate':<8}")
    print("-" * 67)
    for src in sorted(set(r["source"] for r in records)):
        n_pos = counts.get((src, 1), 0)
        n_neg = counts.get((src, 0), 0)
        total = n_pos + n_neg
        print(f"{src:<35} {n_pos:<8} {n_neg:<8} {total:<8} {n_pos/total:.1%}")

    # Split balance
    rec_lookup = {r["question_id"]: r for r in records}
    for split_name, qids in splits.items():
        labels = [rec_lookup[q]["label"] for q in qids]
        pos = sum(labels)
        print(f"\n{split_name}: {len(qids)} records, {pos} syco ({100*pos/len(qids):.1f}%)")


if __name__ == "__main__":
    main()
