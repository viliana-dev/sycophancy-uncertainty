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

### Experiment 2: Continuous Uncertainty Score

**Question:** Does the gradient hold with a continuous measure, not just 3 bins?

**Method:** For each control response, compute uncertainty score:
- Count of hedging phrases in thinking ("however", "on the other hand", "it depends", "not sure", etc.)
- Normalized thinking length (longer deliberation = more uncertain)
- Explicit uncertainty markers ("I'm not sure", "this is debatable")
- Composite z-score

**Analysis:**
- Split into quintiles by uncertainty → sycophancy rate per quintile
- Spearman ρ: uncertainty_score ↔ sycophancy (binary)
- Logistic regression: P(sycophancy) ~ uncertainty_score + source
- Control for source domain (political questions have different base rates)

**Expected:** Monotonic increase. R² shows how much sycophancy is explained by uncertainty alone.

### Experiment 3: Activation Probes Stratified by Uncertainty

**Question:** Can probes detect sycophancy equally well across uncertainty levels?

**Prerequisites:** Run activation extraction (Phase 2) on all 5100 questions.

**Method:**
- Train linear probe on ALL data (sycophantic vs not)
- Evaluate separately per uncertainty quintile
- Compare AUROC across quintiles

**Possible outcomes:**
- Same AUROC everywhere → probe sees one mechanism regardless of uncertainty
- Higher AUROC on uncertain → stronger signal when model "surrenders"
- Lower AUROC on uncertain → uncertain sycophancy is more subtle

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
