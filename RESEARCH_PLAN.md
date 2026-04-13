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

| Layer \ K% | K0 | K10 | K25 | K50 | K75 | K100 |
|---|---|---|---|---|---|---|
| L10 | 0.6990 | 0.7424 | 0.7732 | 0.8268 | 0.8497 | 0.8716 |
| L20 | 0.7815 | 0.8435 | 0.8554 | 0.8771 | 0.8913 | 0.9038 |
| L30 | 0.7898 | 0.8217 | 0.8512 | 0.8837 | 0.8976 | **0.9093** |
| L35 | 0.7875 | 0.8129 | 0.8393 | 0.8800 | 0.8863 | 0.9016 |

K0 = "pre-thinking" baseline: residual stream at the last token before any CoT content, extracted via a single forward pass on `prompt + <think>` (no thinking tokens). Probes trained separately on K0 features.

- **K0 AUROC 0.79 (L30)** — sycophancy signal exists in the prompt representation *before* reasoning begins. The model's residual stream already encodes which records will be sycophantic, at above-chance but far below K100 levels.
- K0→K10 jump is modest (+0.03 at L30), but K10→K100 adds +0.09 — most of the discriminative signal is constructed during CoT, not present at prompt time.
- L20/L30/L35 are close (~0.90); L10 lags (~0.87 max). Best: **L30 K100 = 0.9093**.
- The K0 finding rules out "sycophancy is fully pre-determined by the prompt" — reasoning genuinely contributes to the sycophantic decision — but also rules out "reasoning is the sole source" since 0.79 is well above chance (0.5).

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

Q1−Q5 gap = **0.243** — 4× the heuristic-stratified gap. Effect is dramatic and monotonic across all 24 (layer × K%) configs without exception.

**K=0 pre-thinking stratified by heuristic quintile (L30):**

| Q | Overall | Q1 (conf) | Q2 | Q3 | Q4 | Q5 (unc) |
|---|---|---|---|---|---|---|
| L30_K0 AUROC | 0.790 | **0.857** | 0.796 | 0.693 | 0.789 | 0.785 |

Even before reasoning begins, the probe detects sycophancy at AUROC 0.79. Stratification already shows Q1 > Q5 (0.857 vs 0.785) — the confident/uncertain gap is partially present at prompt time. But the gap is much smaller at K0 than at K100 (0.07 vs 0.24 for entropy), confirming that reasoning amplifies the differential.

**Interpretation:** Sycophancy IS encoded in residual stream activations (overall AUROC 0.91). But the decodability depends sharply on the model's prior confidence:
- **Confident sycophancy** (zero token entropy): probe is essentially perfect (AUROC 0.998). The model "knew" its answer, then deliberately overrode it under persona pressure → sharp activation signature.
- **Uncertain sycophancy** (high token entropy): probe drops to 0.755. The model was already balancing both options; the persona-induced switch looks computationally similar to ordinary deliberation → activations sit near the decision boundary.

This is the *opposite* of the naïve "uncertain → vulnerability with bright neural signal" hypothesis. Uncertainty doesn't amplify the sycophantic signal — it **blurs its decodability**. The two metrics agree on direction but token entropy reveals the effect 4× more strongly, and validates the finding on a principled measure rather than a hand-crafted heuristic.

Two qualitatively distinct sycophancy regimes therefore appear plausible:
1. **Confident override** — internal answer is crisp, sycophancy is a deliberate switch with a clear neural footprint.
2. **Uncertain drift** — internal state is balanced, sycophancy is a smooth tilt toward the persona-favored option indistinguishable in activations from honest reasoning.

A natural follow-up is whether a probe trained only on confident records *transfers* to uncertain ones (Exp 5: cross-prediction) — if not, this is computational evidence for two distinct mechanisms.

### Experiment 4: Probe Confidence Trajectories by Uncertainty Group (DONE)

**Question:** Does the temporal dynamics of sycophancy differ between confident and uncertain responses?

**Method:** Train a single reference probe on L30 K100 (trainval). On the test split, apply it to L30 features at each K ∈ {0, 10, 25, 50, 75, 100} — K0 is the pre-thinking baseline (prompt-only activation at the last token before CoT), K10-K100 are mean-pooled residual at progressively deeper fractions of the CoT. Split test records into 4 groups by (label × confidence regime, confidence = token entropy ≤ trainval median from Exp 5).

**Result — mean P(syco) per group, per K% of CoT:**

| Group | N | K0 | K10 | K25 | K50 | K75 | K100 |
|---|---|---|---|---|---|---|---|
| confident + non-syco | 259 | 0.000 | 0.402 | 0.255 | 0.138 | 0.103 | **0.076** |
| confident + syco     | 133 | 0.000 | 0.660 | 0.717 | 0.783 | 0.822 | **0.825** |
| uncertain + non-syco | 188 | 0.000 | 0.499 | 0.397 | 0.293 | 0.278 | 0.243 |
| uncertain + syco     | 190 | 0.000 | 0.637 | 0.626 | 0.648 | 0.690 | **0.711** |

Note: K0 P(syco) = 0.000 across all groups because the K100-trained probe is out-of-distribution on prompt-only features (different activation subspace). This does NOT mean no signal exists at K0 — Exp 3 shows K0-trained probes achieve AUROC 0.79 (see above). The trajectory P(syco) here reflects the K100 probe's projection, which only activates on thinking-token features.

Syco − non-syco gap per regime:

| K | confident | uncertain |
|---|---|---|
| 0 | +0.000 | +0.000 |
| 10 | +0.258 | +0.138 |
| 25 | +0.462 | +0.229 |
| 50 | +0.646 | +0.354 |
| 75 | +0.718 | +0.412 |
| 100 | **+0.748** | **+0.468** |

**Four distinct trajectory shapes:**

1. **Confident + non-syco: steep descent.** 0.40 → 0.08 — the model starts mildly uncertain, then commits to its honest answer; the longer it reasons, the cleaner the non-sycophantic state becomes. Rich-get-richer for honesty.

2. **Confident + syco: fast commit.** 0.66 → 0.83 — the signal is already huge at K10 (2.6× the confident-nonsyco value at the same depth). The model effectively "decides" to agree in the first 10% of CoT and spends the rest consolidating. This matches the Exp 3/5 finding of a crisp override signature.

3. **Uncertain + non-syco: gentle descent.** 0.50 → 0.24 — same direction as confident-nonsyco but shallower. The model doesn't reach the same clean honest state.

4. **Uncertain + syco: flat drift.** 0.64 → 0.71 — the syco sub-trajectory is almost horizontal (Δ = 0.07 across 90% of CoT). The probe already sees ~0.64 at K10 and barely moves. This is exactly the "drift" interpretation: the activation state is tilted toward the persona-favored option from the start and slowly accumulates a weak signal, rather than committing via a sharp transition.

**Key observations:**

- The sycophancy signal is **present from K10 in both regimes** — decision dynamics are not "deliberate honestly, then surrender at the end". The model's tilt shows up in the first tenth of CoT and the remaining reasoning is mostly post-hoc consolidation.
- **Gap growth confident vs uncertain:** +0.49 vs +0.33 across K10→K100. The confident regime accumulates stronger evidence for sycophancy through CoT; the uncertain regime stays relatively undifferentiated.
- **Confident syco's 0.83 final P(syco) vs uncertain syco's 0.71** directly mirrors the Exp 3 AUROC gap (0.998 vs 0.755): the probe's confidence in sycophancy-labels is itself confidence-regime-dependent.
- **No "late jump" for uncertain syco** — the original hypothesis (deliberate → surrender) is wrong in shape. The actual uncertain pattern is slow tilt, not sharp flip.

This is consistent with the "two mechanisms" reading from Exp 5: confident sycophancy has both an early baseline tilt and an accumulating sharp signature; uncertain sycophancy has only the baseline tilt, which is why its trajectory stays flat and its probe transfers asymmetrically.

### Experiment 5: Cross-Prediction (One Mechanism or Two?) (DONE)

**Question:** Is uncertainty-driven sycophancy the same computational mechanism as confident sycophancy?

**Method:** Split trainval by median uncertainty (computed on trainval only to avoid test leakage). Train two probes on the best (layer, K%) config from Exp 3 (L30, K100): A on the confident half, B on the uncertain half. Evaluate each on both test halves. Run twice: once with the lexical heuristic, once with token entropy.

**Result — HEURISTIC split (median = 6.0):**

| Train→Test | N test | AUROC | Acc |
|---|---|---|---|
| conf→conf | 432 | 0.906 ±0.014 | 0.843 |
| unc→unc | 338 | 0.890 ±0.016 | 0.781 |
| conf→unc | 338 | 0.724 ±0.030 | 0.698 |
| unc→conf | 432 | 0.783 ±0.022 | 0.725 |

Drops: conf→unc −0.18, unc→conf −0.11. Roughly symmetric — both halves lose substantial generalization, so under the lexical metric the picture is blurred.

**Result — TOKEN ENTROPY split (median ≈ 1e-4):**

| Train→Test | N test | AUROC | Acc |
|---|---|---|---|
| conf→conf | 392 | **0.966** ±0.009 | 0.903 |
| unc→unc | 378 | 0.812 ±0.022 | 0.725 |
| conf→unc | 378 | 0.720 ±0.026 | 0.648 |
| unc→conf | 392 | **0.805** ±0.022 | 0.755 |

Drops: conf→unc **−0.247**, unc→conf **−0.007**. The asymmetry is the headline finding. The confident-only probe collapses on uncertain records (−0.25), while the uncertain-only probe transfers to confident records with essentially zero loss (0.805 vs unc-baseline 0.812).

**Interpretation — two mechanisms, one contained in the other:**

1. **Confident sycophancy carries an extra sharp signature** that only exists in the confident regime — a crisp "I knew the answer, now I'm overriding it" direction. The uncertain probe never sees this and therefore never learns it, which is why conf→unc drops so dramatically: the feature that drove conf→conf's 0.966 doesn't exist in uncertain activations.

2. **Uncertain sycophancy's probe captures a more general, weaker direction** that happens to also be present in confident records. Confident sycophancy is so easy that even this weaker direction suffices (0.805 AUROC). But the reverse does not hold — the sharp confident feature is absent when the model was already balancing options, so the confident probe has nothing to grab onto.

This is asymmetric subset-of, not "one mechanism + noise". Confident sycophancy = *general drift* + *sharp override signature*; uncertain sycophancy = *general drift only*. The entropy metric exposes this cleanly because it actually separates the two regimes; the heuristic (more symmetric drops) correlates with uncertainty but doesn't carve the population at its true joint, so the confident-only half still contains many high-entropy cases and vice versa.

**Paper-relevant takeaway:** there are computationally distinct sycophancy pathways, and the confident one is the one that produces the dramatic AUROC 0.998 finding from Exp 3. Any mitigation that targets only the sharp "override" direction (e.g. activation steering in Exp 6) will likely fix confident sycophancy but leave uncertain drift untouched.

---

## Replication: gpt-oss-20b

**Model:** gpt-oss-20b (24 layers, 2880 hidden, MoE 32×4 = 3.6B active params). Harmony chat format with `<|channel|>analysis` (thinking) and `<|channel|>final` (answer). reasoning_effort=medium.

**Data:** Same 5100 questions. 4092 labeled records (80.2% parse rate), 30.9% sycophancy rate (vs Qwen's 41.9%). 1008 skipped: essay-style answers ("The answer depends on...") and refusals — gpt-oss is less formulaic in answer formatting.

### Exp 3 gptoss: Probes by Uncertainty

**Best config: L12_K100, AUROC = 0.9091** — identical to Qwen's 0.9093.

Probe sweep (test AUROC):

| Layer \ K% | K10 | K25 | K50 | K75 | K100 |
|---|---|---|---|---|---|
| L6 | 0.767 | 0.767 | 0.821 | 0.848 | 0.864 |
| L12 | 0.805 | 0.820 | 0.857 | 0.872 | **0.909** |
| L18 | 0.811 | 0.836 | 0.843 | 0.867 | 0.908 |
| L21 | 0.796 | 0.812 | 0.836 | 0.857 | 0.897 |

Same monotonic K-improvement, similar layer pattern (mid-layers best).

**K0 pre-thinking (probe trained on K0 features separately):**

| Layer | K0 AUROC |
|---|---|
| L6 | 0.772 |
| L12 | 0.776 |
| L18 | **0.810** |
| L21 | 0.795 |

**K0 AUROC = 0.81 (L18)** — comparable to Qwen (0.79 at L30). Sycophancy signal exists before reasoning in both architectures.

K0 by entropy quintile (L18): Q1=0.874, Q2=0.819, Q3=0.787, Q4=0.706, Q5=0.774. Mild confident→uncertain gradient at K0 (same direction as Qwen but weaker).

**Entropy-stratified quintiles (rank-based, L12 K100):**

| Q | N | Pos% | AUROC |
|---|---|---|---|
| Q1 (lowest entropy) | 124 | 15.3% | 0.901 |
| Q2 | 124 | 21.8% | 0.918 |
| Q3 | 124 | 32.3% | 0.922 |
| Q4 | 124 | 25.8% | 0.868 |
| Q5 (highest entropy) | 124 | 59.7% | 0.890 |

**Q1−Q5 gap = 0.011** — essentially flat. This is the KEY difference from Qwen (gap = 0.243). gpt-oss does NOT show the confident→uncertain AUROC degradation that Qwen shows. The probe works equally well across all entropy quintiles.

**Interpretation:** gpt-oss appears to use a single, more uniform sycophancy mechanism that doesn't vary with uncertainty. The MoE architecture and/or RLHF training may produce a more homogeneous representation — sycophancy "looks the same" whether the model was confident or not.

### Exp 4 gptoss: Trajectories (token-entropy grouping)

| Group | N | K10 | K25 | K50 | K75 | K100 |
|---|---|---|---|---|---|---|
| confident + non-syco | 258 | 0.258 | 0.182 | 0.139 | 0.119 | 0.090 |
| confident + syco | 74 | 0.500 | 0.496 | 0.548 | 0.674 | 0.741 |
| uncertain + non-syco | 170 | 0.464 | 0.318 | 0.197 | 0.172 | 0.151 |
| uncertain + syco | 118 | 0.639 | 0.613 | 0.677 | 0.684 | 0.721 |

Same 4 trajectory shapes as Qwen: steep descent (conf-nonsyco), fast commit (conf-syco), gentle descent (unc-nonsyco), flat drift (unc-syco). The uncertain-syco trajectory is again nearly horizontal (0.64→0.72, Δ=0.08). Despite the flat Exp 3 quintile gradient, the trajectory shapes replicate.

Syco−nonsyco gap at K100: confident **+0.651**, uncertain **+0.570** (Qwen: +0.748 / +0.468). The gap ratio is smaller (1.14× vs 1.60× for Qwen) — consistent with more uniform mechanism.

### Exp 5 gptoss: Cross-Prediction

**Heuristic split:**

| Train→Test | AUROC | Drop |
|---|---|---|
| conf→conf | 0.954 | — |
| unc→unc | 0.888 | — |
| conf→unc | 0.704 | −0.250 |
| unc→conf | 0.659 | −0.229 |

**Entropy split:**

| Train→Test | AUROC | Drop |
|---|---|---|
| conf→conf | 0.934 | — |
| unc→unc | 0.910 | — |
| conf→unc | 0.635 | **−0.299** |
| unc→conf | 0.687 | **−0.223** |

Both directions lose ~0.25. This is SYMMETRIC — unlike Qwen where unc→conf had essentially zero drop (−0.007). In gpt-oss, the uncertain-trained probe does NOT transfer to confident records, meaning the uncertain regime captures different features that aren't present in the confident regime either.

**Interpretation:** gpt-oss shows **mutual non-transferability** rather than Qwen's **asymmetric containment**. Neither regime's probe transfers well to the other. This could mean gpt-oss has two genuinely distinct mechanisms (vs Qwen's "one contained in the other"), or that the MoE routing creates different feature subspaces for different uncertainty levels.

### Cross-Model Comparison Summary

| Finding | Qwen3-14B | gpt-oss-20b |
|---|---|---|
| Best AUROC | 0.909 (L30 K100) | 0.909 (L12 K100) |
| K0 pre-thinking AUROC | 0.790 (L30) | 0.810 (L18) |
| Sycophancy rate | 41.9% | 30.9% |
| Entropy Q1−Q5 AUROC gap | **0.243** (dramatic) | **0.011** (flat) |
| Trajectory shapes | 4 distinct ✓ | 4 distinct ✓ |
| Cross-prediction (entropy) | asymmetric (conf⊃unc) | symmetric (mutual drop) |
| Uncertain syco trajectory | flat drift ✓ | flat drift ✓ |

**What replicates:** Overall probe accuracy (0.91), 4 trajectory shapes, uncertain-syco flat drift, sycophancy signal from K10.

**What differs:** Qwen shows sharp confident/uncertain bifurcation in probe quality and asymmetric transfer; gpt-oss shows uniform probe quality and symmetric non-transfer. The sycophancy-uncertainty relationship is architecture-dependent — but the core finding (sycophancy is detectable and mechanistically different across uncertainty levels) holds in both.

---

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
