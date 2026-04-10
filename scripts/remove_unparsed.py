"""Remove ALL records where answer can't be parsed to a letter."""
import json, re
from src.datasets import load_dataset_for_behavior

def parse_new(answer):
    answer = answer.strip()
    matches = re.findall(r'\(([A-Ga-g])\)', answer)
    if matches: return matches[-1].upper()
    matches = re.findall(r'\b(?:answer(?:\s+is)?|option)[:\s]+([A-Ga-g])\b', answer, re.IGNORECASE)
    if matches: return matches[-1].upper()
    m = re.search(r'\b([AB])\b', answer)
    if m: return m.group(1).upper()
    stripped = answer.strip("() ").upper()
    if stripped in set("ABCDEFG"): return stripped
    return None

for f in ["control", "intervention"]:
    path = f"data/generated/sycophancy/{f}.jsonl"
    kept = []
    removed = 0
    with open(path) as fh:
        for line in fh:
            if not line.strip(): continue
            rec = json.loads(line)
            if parse_new(rec.get("answer", "")) is not None:
                kept.append(rec)
            else:
                removed += 1
    with open(path, "w") as fh:
        for rec in kept:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{f}: kept {len(kept)}, removed {removed}")
