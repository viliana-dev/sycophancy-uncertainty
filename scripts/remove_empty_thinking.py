"""Remove records with empty thinking (no CoT for probes)."""
import json

for f in ["control", "intervention"]:
    path = f"data/generated/sycophancy/{f}.jsonl"
    kept = []
    removed = 0
    with open(path) as fh:
        for line in fh:
            if not line.strip(): continue
            rec = json.loads(line)
            if not rec.get("thinking", "").strip():
                removed += 1
            else:
                kept.append(rec)
    with open(path, "w") as fh:
        for rec in kept:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{f}: kept {len(kept)}, removed {removed} empty thinking")
