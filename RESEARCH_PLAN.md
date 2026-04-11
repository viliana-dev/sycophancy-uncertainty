# Epistemic Uncertainty as Vulnerability to Sycophancy in Reasoning Models

**Target:** EIML Workshop @ ICML 2026 (4-6 pages)
**Authors:** Viliana Devbunova
**Subject model:** Qwen3-14B
**Dataset:** Anthropic sycophancy evals, 5100 questions (1700 per source, stratified)

---

## Core Claim

LLMs don't say "I don't know" — they resolve epistemic uncertainty by agreeing with the user. We show quantitatively that model uncertainty predicts sycophantic behavior: the less confident a model is in its own answer, the more likely it is to flip to the persona's opinion.

This reframes sycophancy from "social deference" to **epistemic shortcut** — and opens a new avenue for mitigation through uncertainty-aware decoding.

## Background

Existing sycophancy research either:
- Studies factual sycophancy where model knows the answer (MONICA, "When Truth Is Overridden")
- Filters out uncertain cases to study "pure" sycophancy ("Sycophancy Is Not One Thing")
- Treats sycophancy as a single phenomenon without stratifying by confidence

**Nobody has shown:** uncertainty ↔ sycophancy correlation quantitatively on a large dataset.

## Data Collection (DONE)

### Dataset
- Anthropic sycophancy evals: 3 domains × 1700 questions = 5100 total
  - `sycophancy_on_nlp_survey` (NLP research opinions)
  - `sycophancy_on_philpapers2020` (philosophy positions)
  - `sycophancy_on_political_typology_quiz` (political attitudes)

### Generation
- **Control pass:** question without persona → model's "natural" answer
- **Intervention pass:** question with persona bio → model's influenced answer
- Model: Qwen3-14B, local inference (greedy, do_sample=False), 4×A800-80GB
- Full thinking (CoT) and answer saved for both passes

### Parser
- Supports (A)-(G) answers + `\boxed{}` format
- Takes LAST matched letter (handles self-corrections)
- Parse rate: 99.0% control, 99.9% intervention, 98.9% both

### Labeling
- Sycophantic = intervention answer matches `answer_matching_behavior` AND differs from control
- No external LLM judge needed — purely mechanical

## Experiments

### Experiment 1: Uncertainty Gradient (DONE)

**Question:** Does model uncertainty in control predict sycophancy rate in intervention?

**Method:** Classify control responses by uncertainty level:
- **Hard refusal** (N=51): model didn't produce a parseable answer ("As an AI, I don't have opinions")
- **Strict soft refusal** (N=399): model answered but included refusal language in its answer text
- **Clean answer** (N=4277): model answered without any refusal/hedging

**Result:**

| Category | N | Sycophancy rate |
|---|---|---|
| Hard refusal | 51 | **89.8%** |
| Strict soft refusal | 399 | **49.6%** |
| Clean answer | 4277 | **39.7%** |

Monotonic gradient: more uncertain → more sycophantic. Hard refusals are 2.3× more sycophantic than clean answers.

### Experiment 2: Continuous Uncertainty Score (DONE)

**Question:** Does the gradient hold with a continuous measure, not just 3 bins?

**Method:** For each control response, compute uncertainty score (4 components):
- Hedging phrase count in thinking (24 patterns: "however", "on the other hand", "it depends", "not sure", etc.)
- AI refusal phrases in thinking (weighted ×2)
- AI refusal phrases in answer (weighted ×3 — strongest signal)
- Normalized thinking length (longer deliberation = more uncertain)

**Analysis:** N=5096 usable records, 41.9% overall sycophancy rate.

**Result — Quintiles (monotonic trend):**

| Quintile | Uncertainty | Sycophancy rate |
|---|---|---|
| Q1 (lowest) | low | **34.8%** |
| Q2 | | 39.3% |
| Q3 | | 44.1% |
| Q4 | | 43.7% |
| Q5 (highest) | high | **46.7%** |

Difference Q1→Q5: +11.9 p.p.

**Result — Deciles (finer gradient):**

| Decile | Sycophancy rate |
|---|---|
| D1 (most confident) | **31.5%** |
| D10 (most uncertain) | **51.0%** |

Difference D1→D10: +19.5 p.p., nearly 2×. D7 (33.8%) breaks strict monotonicity.

**Correlation:**
- Spearman ρ = 0.093, p = 2.85e-11 — statistically significant but weak effect
- Logistic regression coeff = +0.19 (positive = more uncertain → more syco)
- Effect persists after controlling for source domain (coeff = +0.15)

**By source (correlation significant in each separately):**
- philpapers: 47.1% sycophancy (highest)
- political: 42.7%
- nlp_survey: 35.9% (lowest)

**Honest assessment:** Trend is real and statistically significant, but ρ=0.093 is a weak effect — uncertainty explains ~1% of sycophancy variance. Main predictor is source domain, not uncertainty. However, extreme cases remain compelling: hard refusal (89.8%) vs clean answer (35.7%) = 2.5× difference. Continuous metric shows a weak but significant gradient in between.

### Experiment 3: Activation Probes Stratified by Uncertainty

**Question:** Can probes detect sycophancy equally well across uncertainty levels?

**Setup:** Linear probes (StandardScaler + LogisticRegression, C=1) on mean-pooled residual stream activations from Qwen3-14B intervention responses. Sweep over layers [10, 20, 30, 35] × thinking truncation K% [10, 25, 50, 75, 100]. 5096 records, stratified split 70/15/15 (train=3565, val=761, test=770).

**Probe sweep — Test AUROC by (layer × K%):**

| Layer \ K% | K10 | K25 | K50 | K75 | K100 |
|---|---|---|---|---|---|
| L10 | 0.7424 | 0.7732 | 0.8268 | 0.8497 | 0.8716 |
| L20 | 0.8435 | 0.8554 | 0.8771 | 0.8913 | 0.9038 |
| L30 | 0.8217 | 0.8512 | 0.8837 | 0.8976 | **0.9093** |
| L35 | 0.8129 | 0.8393 | 0.8800 | 0.8863 | 0.9016 |

- AUROC monotonically increases with K% across all layers → sycophancy crystallizes towards the end of reasoning, not in the first tokens.
- L20/L30/L35 are close (~0.90); L10 lags (~0.87 max). Best: **L30 K100 = 0.9093**.

**Stratification by uncertainty — two metrics:**

1. **Lexical heuristic** (hand-crafted hedging + AI-refusal phrase counts in control thinking, weighted by length). Captures *semantic* deliberation.
2. **Token entropy** (`compute_answer_entropy.py`): a single forward pass per control record building `prompt + <think>thinking</think>\n\nThe answer is (`, then softmax over option-letter logits at the next-token position. Principled, model-derived signal.

**Distribution of token entropy:** heavily bimodal — 65% of records have entropy < 0.001 (RLHF makes Qwen3-14B output near-deterministic letter logits even when it deliberated extensively in CoT). Only the top 35% has any meaningful entropy mass. This is itself an interesting negative result: for reasoning models, lexical signals from the CoT carry richer uncertainty information than logit entropy at the answer position. Both metrics are still tested.

**Exp 2 redo with token entropy:** Spearman ρ = 0.144 (vs 0.093 for the heuristic), logistic coef +1.28 (vs +0.19). Token entropy is the *stronger* sycophancy predictor of the two — zero-entropy records have 37.1% sycophancy rate, nonzero have 50.9% (+14 pp).

**Best probe (L30 K100) stratified by HEURISTIC quintile (percentile edges):**

| Q | Range | N | Pos% | AUROC | Acc |
|---|---|---|---|---|---|
| Q1 (most confident) | 1.2-4.4 | 150 | 35.3% | **0.949** ±0.016 | 0.860 |
| Q2 | 4.4-5.5 | 158 | 33.5% | 0.918 ±0.021 | 0.829 |
| Q3 | 5.5-6.7 | 149 | 49.0% | 0.893 ±0.026 | 0.799 |
| Q4 | 6.7-8.0 | 118 | 42.4% | 0.893 ±0.030 | 0.814 |
| Q5 (most uncertain) | 8.0-18.0 | 195 | 48.2% | **0.888** ±0.023 | 0.800 |

Q1−Q5 gap = 0.061 (~3 SE). Monotonic trend across nearly all 20 (layer × K%) configs.

**Best probe (L30 K100) stratified by TOKEN ENTROPY quintile (rank-based):**

| Q | Range | N | Pos% | AUROC | Acc |
|---|---|---|---|---|---|
| Q1 (zero entropy, confident) | 0.0 | 154 | 31.8% | **0.998** ±0.002 | 0.974 |
| Q2 | 0.0 | 154 | 34.4% | 0.949 ±0.017 | 0.870 |
| Q3 | 0.0 | 154 | 39.6% | 0.890 ±0.026 | 0.799 |
| Q4 | 0.0 | 154 | 49.4% | 0.854 ±0.030 | 0.766 |
| Q5 (nonzero, uncertain) | 0.0–0.7 | 154 | 54.5% | **0.755** ±0.039 | 0.688 |

Q1−Q5 gap = **0.243** — 4× the heuristic-stratified gap. Effect is dramatic and monotonic across all 20 (layer × K%) configs without exception.

**Interpretation:** Sycophancy IS encoded in residual stream activations (overall AUROC 0.91). But the decodability depends sharply on the model's prior confidence:
- **Confident sycophancy** (zero token entropy): probe is essentially perfect (AUROC 0.998). The model "knew" its answer, then deliberately overrode it under persona pressure → sharp activation signature.
- **Uncertain sycophancy** (high token entropy): probe drops to 0.755. The model was already balancing both options; the persona-induced switch looks computationally similar to ordinary deliberation → activations sit near the decision boundary.

This is the *opposite* of the naïve "uncertain → vulnerability with bright neural signal" hypothesis. Uncertainty doesn't amplify the sycophantic signal — it **blurs its decodability**. The two metrics agree on direction but token entropy reveals the effect 4× more strongly, and validates the finding on a principled measure rather than a hand-crafted heuristic.

Two qualitatively distinct sycophancy regimes therefore appear plausible:
1. **Confident override** — internal answer is crisp, sycophancy is a deliberate switch with a clear neural footprint.
2. **Uncertain drift** — internal state is balanced, sycophancy is a smooth tilt toward the persona-favored option indistinguishable in activations from honest reasoning.

A natural follow-up is whether a probe trained only on confident records *transfers* to uncertain ones (Exp 5: cross-prediction) — if not, this is computational evidence for two distinct mechanisms.

### Experiment 4: Probe Confidence Trajectories by Uncertainty Group

**Question:** Does the temporal dynamics of sycophancy differ between confident and uncertain responses?

**Prerequisites:** Activation extraction + trained probe from Exp 3.

**Method:** Plot P(sycophancy) token-by-token through CoT for:
- Confident sycophancy (clean control + flipped to persona)
- Uncertain sycophancy (soft refusal + flipped to persona)
- Non-sycophantic control

**Hypothesis:**
- Confident: gradual rise from start (model "knows" it will agree)
- Uncertain: flat then sharp jump (model deliberates then "surrenders")
- Control: flat low

Different trajectory shapes = different decision dynamics = publishable finding.

### Experiment 5: Cross-Prediction (One Mechanism or Two?)

**Question:** Is uncertainty-driven sycophancy the same computational mechanism as confident sycophancy?

**Method:**
- Probe A: trained only on confident samples (uncertainty < median)
- Probe B: trained only on uncertain samples (uncertainty > median)
- Cross-evaluate: A on uncertain test set, B on confident test set
- Compare with cosine similarity of weight vectors

**Interpretation:**
- High cross-AUROC + high cosine → one mechanism, uncertainty just amplifies it
- Low cross-AUROC + low cosine → two distinct pathways

### Experiment 6 (Stretch): Intervention Restores Uncertainty Expression

**Question:** Can we reverse sycophancy by subtracting the "agreement direction" from activations?

**Method:**
- Compute sycophancy direction from probe weight vector
- During intervention-prompt generation, subtract direction from residual stream at layer L
- Check: does the model stop agreeing with persona?
- On uncertain questions: does refusal ("As an AI...") come back?

**Strong result:** Intervention restores principled refusal = "we can programmatically repair safety behavior suppressed by persona pressure"

## Timeline

| Week | Tasks |
|---|---|
| Week 1 | Exp 2 (continuous uncertainty), start activation extraction |
| Week 2 | Exp 3 + 4 (probes + trajectories) |
| Week 3 | Exp 5 (cross-prediction), start writing |
| Week 4 | Exp 6 if time, finish paper |

## Key Differentiators from Prior Work

1. **Quantitative uncertainty→sycophancy link** — nobody has shown this
2. **N=5100** — largest sycophancy probe study (MONICA: 500 train, scheming-probes v1: 649)
3. **Temporal dynamics** — trajectory analysis shows WHEN model decides to defer
4. **Refusal override** — persona breaks safety training (100% on political questions)
5. **Epistemic framing** — sycophancy as failure of uncertainty expression, not just "social deference"

## References

- Sharma et al. 2024 — "Towards Understanding Sycophancy in Language Models" (ICLR 2024)
- Wen et al. 2025 — "Sycophancy Is Not One Thing: Causal Separation" (ICLR 2026)
- Hu et al. 2025 — "MONICA: Real-Time Monitoring and Calibration" (rejected ICLR 2026)
- Luo et al. 2025 — "When Truth Is Overridden" (mechanistic origins)
- Kramár et al. 2026 — "Building Production-Ready Probes for Gemini"
