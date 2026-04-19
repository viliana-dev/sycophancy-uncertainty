"""Filter out records where control answer could not be parsed.

Backs up original files, writes filtered labeled.jsonl and new splits.json.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import GENERATED_DIR, TRAIN_SPLIT, VAL_SPLIT


MODELS = {
    "sycophancy": {"ctrl_field": "ctrl_answer"},
    "sycophancy_gptoss": {"ctrl_field": "parsed_control_answer"},
    "sycophancy_qwen_moe": {"ctrl_field": "parsed_control_answer"},
}


def main():
    for model_dir, cfg in MODELS.items():
        gen_dir = GENERATED_DIR / model_dir
        labeled_path = gen_dir / "labeled.jsonl"
        if not labeled_path.exists():
            print(f"Skip {model_dir}: no labeled.jsonl")
            continue

        with open(labeled_path) as f:
            records = [json.loads(l) for l in f]

        ctrl_field = cfg["ctrl_field"]
        before = len(records)
        clean = [r for r in records if r.get(ctrl_field) is not None]
        dropped = before - len(clean)

        syco_before = sum(1 for r in records if r["label"] == 1)
        syco_after = sum(1 for r in clean if r["label"] == 1)

        print(f"\n=== {model_dir} ===")
        print(f"Before: {before} (syco={syco_before}, {100*syco_before/before:.1f}%)")
        print(f"Dropped: {dropped} (ctrl=None)")
        print(f"After:  {len(clean)} (syco={syco_after}, {100*syco_after/len(clean):.1f}%)")

        # Backup originals
        backup_l = gen_dir / "labeled_unfiltered.jsonl"
        backup_s = gen_dir / "splits_unfiltered.json"
        if not backup_l.exists():
            labeled_path.rename(backup_l)
            print(f"Backed up -> {backup_l.name}")
        else:
            print(f"Backup already exists, overwriting labeled.jsonl")

        splits_path = gen_dir / "splits.json"
        if splits_path.exists() and not backup_s.exists():
            splits_path.rename(backup_s)

        # Write filtered
        with open(labeled_path, "w") as f:
            for r in clean:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Recreate splits (same logic as prepare_labeled)
        rng = random.Random(42)
        groups = {}
        for r in clean:
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
        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")

        # Summary by source
        counts = Counter((r["source"], r["label"]) for r in clean)
        print(f"\n{'Source':<40} {'Syco':<6} {'Not':<6} {'Rate':<6}")
        print("-" * 58)
        for src in sorted(set(r["source"] for r in clean)):
            n_p = counts.get((src, 1), 0)
            n_n = counts.get((src, 0), 0)
            print(f"{src:<40} {n_p:<6} {n_n:<6} {n_p/(n_p+n_n):.1%}")


if __name__ == "__main__":
    main()
