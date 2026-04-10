"""Check: when model refuses in control, does it answer with persona?"""
import json, re
from collections import Counter
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

target_qs = load_dataset_for_behavior("sycophancy", n_samples=5100, seed=42)
target_ids = {q.question_id for q in target_qs}
q_lookup = {q.question_id: q for q in target_qs}

# Load all records
ctrl = {}
intv = {}
for f, store in [("control", ctrl), ("intervention", intv)]:
    with open(f"data/generated/sycophancy/{f}.jsonl") as fh:
        for line in fh:
            if not line.strip(): continue
            rec = json.loads(line)
            if rec["question_id"] in target_ids:
                store[rec["question_id"]] = rec

# Find questions where control has no parsed answer
ctrl_no_parse = {qid for qid, rec in ctrl.items() if parse_new(rec.get("answer", "")) is None}
print(f"Control: {len(ctrl_no_parse)} unparsed out of {len(ctrl)}")

# Of those, how many have a parsed intervention answer?
flip_to_answer = []
still_no_answer = []
for qid in ctrl_no_parse:
    irec = intv.get(qid)
    if not irec:
        continue
    i_parsed = parse_new(irec.get("answer", ""))
    q = q_lookup.get(qid)
    amb = q.answer_matching_behavior.strip() if q else ""

    if i_parsed is not None:
        flip_to_answer.append({
            "qid": qid,
            "source": ctrl[qid].get("source", ""),
            "ctrl_answer": ctrl[qid].get("answer", "")[:200],
            "intv_answer": irec.get("answer", "")[:200],
            "intv_parsed": i_parsed,
            "sycophantic_letter": amb,
            "is_sycophantic": i_parsed == amb.strip("() "),
            "q_ctrl": q.question_control[:150] if q else "",
        })
    else:
        still_no_answer.append(qid)

print(f"\nWith persona -> answered: {len(flip_to_answer)}")
print(f"With persona -> still no answer: {len(still_no_answer)}")

# Of those that flipped, how many agreed with persona (sycophantic)?
syco = sum(1 for r in flip_to_answer if r["is_sycophantic"])
not_syco = sum(1 for r in flip_to_answer if not r["is_sycophantic"])
print(f"\nOf {len(flip_to_answer)} that flipped to answering:")
print(f"  Agreed with persona (sycophantic): {syco} ({100*syco/max(1,len(flip_to_answer)):.1f}%)")
print(f"  Disagreed with persona:            {not_syco} ({100*not_syco/max(1,len(flip_to_answer)):.1f}%)")

# By source
print("\nFlips by source:")
by_src = Counter(r["source"] for r in flip_to_answer)
for src, cnt in by_src.most_common():
    src_syco = sum(1 for r in flip_to_answer if r["source"] == src and r["is_sycophantic"])
    print(f"  {src}: {cnt} flipped, {src_syco} sycophantic ({100*src_syco/cnt:.0f}%)")

# Show examples
print("\n" + "=" * 80)
print("EXAMPLES: refused in control -> answered with persona")
print("=" * 80)
for i, r in enumerate(flip_to_answer[:10]):
    tag = "SYCOPHANTIC" if r["is_sycophantic"] else "independent"
    print(f"\n--- [{i+1}] {r['qid']} ({r['source']}) [{tag}] ---")
    print(f"Q: {r['q_ctrl'][:120]}")
    print(f"Sycophantic answer: {r['sycophantic_letter']}, Model chose: {r['intv_parsed']}")
    print(f"CONTROL: {r['ctrl_answer'][:180]}")
    print(f"INTERVENTION: {r['intv_answer'][:180]}")
