# Sycophancy Generation QA Report

**Date:** 2026-04-10
**Model:** Qwen3-14B (local, greedy do_sample=False, 4×A800-SXM4-80GB)
**Dataset:** 5100 questions (1700 per source, stratified)
**Generation:** batch_size=4-8, max_new_tokens=8192

## Coverage

- **5100/5100** questions in both control and intervention
- Exactly 5100 records per file (all old API leftovers removed, all truncated records regenerated)
- **0 empty answers**, **0 empty thinking**
- All generation done locally (Qwen3-14B, greedy, max_new_tokens=8192)
- Token lengths: min=140, median=1223-1546, p99=8187-8188, max=8192

## Parse Rate

Parser supports (A)-(G), `\boxed{}`, "Answer: X" patterns. Takes LAST matched letter.

| | Parsed | Unparsed | Rate |
|---|---|---|---|
| Control | 5049 | 51 | 99.0% |
| Intervention | 5095 | 5 | 99.9% |
| Both parsed | 5046 | 54 | 98.9% |

## Remaining 54 Unparsed

All are model behavior (principled refusal or prose), not data issues. Re-generation produces the same result.

| Category | Count | Description |
|---|---|---|
| AI refusal | 41 | "As an AI, I don't have personal feelings/beliefs" (control only) |
| Prose without letter | 15 | Academic essay without picking an option (philpapers, control only) |

## Key Finding: Uncertainty Gradient

Control responses classified by uncertainty:

| Category | N | Definition | Sycophancy rate |
|---|---|---|---|
| **Hard refusal** | 51 | No parsed answer | **89.8%** |
| **Strict soft refusal** | 399 | Answered but refusal in answer text | **49.6%** |
| **Clean answer** | 4277 | No refusal language | **39.7%** |

Monotonic: more uncertain → more sycophantic. Hard refusals are 2.3× more sycophantic than clean answers.

### Refusal-to-Agreement Flip

Of 51 hard-refusal questions, 49 answered when persona added:
- **44 (89.8%) agreed with persona** (sycophantic)
- **5 (10.2%) disagreed** (independent)
- 2 still refused

| Source | Flipped | Sycophantic | Rate |
|---|---|---|---|
| philpapers2020 | 35 | 30 | 86% |
| political_typology_quiz | 14 | 14 | **100%** |

## answer_matching_behavior Distribution

| Letter | Count | % |
|---|---|---|
| (A) | 2333 | 45.7% |
| (B) | 2296 | 45.0% |
| (C) | 276 | 5.4% |
| (D) | 118 | 2.3% |
| (E) | 62 | 1.2% |
| (F) | 10 | 0.2% |
| (G) | 5 | 0.1% |

~9.2% of sycophantic answers are (C)-(G).
