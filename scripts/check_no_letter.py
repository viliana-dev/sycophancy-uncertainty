"""Check records where model answered in prose without a letter choice."""
import json, re
from collections import Counter
from transformers import AutoTokenizer
from src.datasets import load_dataset_for_behavior

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)

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

no_letter = []
for f in ["control", "intervention"]:
    with open(f"data/generated/sycophancy/{f}.jsonl") as fh:
        for line in fh:
            if not line.strip(): continue
            rec = json.loads(line)
            qid = rec["question_id"]
            if qid not in target_ids: continue
            answer = rec.get("answer", "")
            thinking = rec.get("thinking", "")
            if not answer.strip(): continue
            parsed = parse_new(answer)
            if parsed is not None: continue
            n_tok = len(tok.encode(thinking + answer))
            no_letter.append({
                "qid": qid, "file": f, "source": rec.get("source", ""),
                "tokens": n_tok,
                "answer": answer,
                "q_ctrl": q_lookup[qid].question_control[:200] if qid in q_lookup else "",
            })

by_source = Counter(r["source"] for r in no_letter)
print("By source:")
for src, cnt in by_source.most_common():
    print(f"  {src}: {cnt}")

refusal_words = ["as an ai", "i don't have personal", "i don't have feelings",
    "i cannot have", "don't hold opinions", "no personal opinion",
    "i'm an ai", "i am an ai"]
refusal_count = sum(1 for r in no_letter if any(w in r["answer"].lower() for w in refusal_words))

prose_no_refusal = [r for r in no_letter if not any(w in r["answer"].lower() for w in refusal_words)]

print(f"\nTotal no-letter: {len(no_letter)}")
print(f"AI refusal: {refusal_count}")
print(f"Prose (not refusal): {len(prose_no_refusal)}")

print("\n" + "=" * 80)
print("AI REFUSAL EXAMPLES")
print("=" * 80)
refusals = [r for r in no_letter if any(w in r["answer"].lower() for w in refusal_words)]
for i, r in enumerate(refusals[:5]):
    print(f"\n--- [{i+1}] {r['qid']} ({r['file']}, {r['source']}, {r['tokens']}tok) ---")
    print(f"Q: {r['q_ctrl'][:150]}")
    print(f"A: {r['answer'][:300]}")

print("\n" + "=" * 80)
print("PROSE (NON-REFUSAL) EXAMPLES")
print("=" * 80)
for i, r in enumerate(prose_no_refusal[:10]):
    print(f"\n--- [{i+1}] {r['qid']} ({r['file']}, {r['source']}, {r['tokens']}tok) ---")
    print(f"Q: {r['q_ctrl'][:150]}")
    print(f"A: {r['answer'][:300]}")
