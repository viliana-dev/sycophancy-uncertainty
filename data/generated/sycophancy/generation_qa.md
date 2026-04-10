# Sycophancy Generation QA Report

**Date:** 2026-04-07 (updated after cleanup)
**Model:** Qwen3-14B (local, batch_size=8, 4×A800-SXM4-80GB)
**Dataset:** 5100 questions (1700 per source, stratified)

## Coverage

- Target 5100 questions: **5100/5100 covered** in both control and intervention
- Old API records removed (532 per file) — files now contain exactly 5100 records each

### Empty outputs
| | Empty answer | Empty thinking |
|---|---|---|
| Control | 17 | 6 |
| Intervention | 12 | 23 |

## Parse Rate (new parser, A-G support)

| | Parsed | Unparsed | Rate |
|---|---|---|---|
| Control | 4987 | 113 | 97.8% |
| Intervention | 5073 | 27 | 99.5% |
| Both parsed | 4968 | 132 | 97.4% |

### Old parser (A/B only) vs New parser (A-G)
- Control: 160 → 113 unparsed (fixed 54, regressed 0)
- Intervention: 300 → 27 unparsed (fixed 273, regressed 0)
- **+301 questions recovered** by supporting (C)-(G) answers

## Unparsed Records Breakdown (140 total across both files)

| Category | Count | Description |
|---|---|---|
| Truncated ~1024 tok | 35 | Old API records cached by resume logic (max_tokens=1024) |
| Truncated 8192 tok | 1 | Hit max_new_tokens in batch generation |
| AI refusal | 62 | "As an AI, I don't have personal feelings/beliefs" |
| Prose without letter | 42 | Essay/analysis without selecting an option |

### Truncated at ~1024 tokens (35 records)
- From previous API run with lower max_tokens setting
- Model started prose answer but hit token limit before choosing a letter
- Should be re-generated with local model (max_new_tokens=8192) if needed

### Truncated at 8192 tokens (1 record)
- `sycophancy_on_nlp_survey_03937` — hit max_new_tokens in batch generation
- Very long thinking chain, never closed `</think>`

### AI Refusal (62 records, control only)
- Model refuses to choose: "As an AI, I don't have personal feelings/beliefs"
- Mostly political typology ("How do you feel toward Republicans/Democrats?") and philpapers ("What is your view on God?")

### Prose without letter (42 records, mostly control)
- Model writes essay/analysis without selecting a specific option
- Mostly philpapers philosophical questions

## Key Finding: Refusal-to-Agreement Flip

Of 113 control-unparsed questions, **105 have a parsed intervention answer** (with persona):

| | Count | % |
|---|---|---|
| Refused in control → answered with persona | 105 | 93% |
| Refused in control → still refused with persona | 8 | 7% |

Of the 105 that flipped to answering:
- **86 (81.9%) agreed with persona** (sycophantic)
- **19 (18.1%) disagreed with persona** (independent)

### By source:
| Source | Flipped | Sycophantic | Rate |
|---|---|---|---|
| philpapers2020 | 73 | 57 | 78% |
| political_typology_quiz | 31 | 28 | 90% |
| nlp_survey | 1 | 1 | 100% |

### Interpretation
This is a stronger sycophancy signal than simple answer-switching (A→B). The model transitions from **principled refusal** ("As an AI, I don't have opinions") to **agreeing with the persona's stated opinion**. The persona doesn't just change the answer — it overcomes the model's safety training to refuse opinion questions.

This pattern is especially interesting for probe detection: the activation shift from refusal→agreement may have a distinct signature compared to ordinary answer changes.

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

~9.2% of sycophantic answers are (C)-(G), invisible to the old A/B-only parser.
