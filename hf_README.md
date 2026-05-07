---
license: cc-by-4.0
task_categories:
  - text-classification
language:
  - en
tags:
  - sycophancy
  - interpretability
  - activation-probing
  - mixture-of-experts
  - epistemic-uncertainty
pretty_name: Nested Geometry of Sycophancy
size_categories:
  - 10K<n<100K
---

# Nested Geometry of Sycophancy

Dataset and activation features for the paper **"The Nested Geometry of Epistemic Capitulation: Asymmetric Transfer of Sycophancy Mechanisms Across LLM Architectures"** (EIML @ ICML 2026).

## Overview

We study the internal geometry of sycophancy in reasoning LLMs, stratified by epistemic uncertainty. The dataset contains:

- **Sycophancy evaluation data** from Anthropic model-written evaluations, with model-generated Chain-of-Thought responses and sycophancy labels
- **Activation features** (residual stream) extracted at multiple layers and CoT depth percentages
- **Token-level entropy** measuring model uncertainty
- **Steering experiment results** testing static linear interventions

Three models are covered:

| Model | Type | Layers | Hidden dim | Best layer | Suffix |
|---|---|---|---|---|---|
| Qwen3-14B | Dense | 40 | 5120 | L30 | `sycophancy` |
| gpt-oss-20b | MoE (32x4) | 24 | 2880 | L18 | `sycophancy_gptoss` |
| Qwen3-30B-A3B | MoE (128x8) | 48 | 2048 | L36 | `sycophancy_qwen_moe` |

## File structure

```
data/
  generated/
    sycophancy{_gptoss,_qwen_moe}/
      labeled.jsonl          # question_id, question, answer, CoT, label (0/1), uncertainty_score
      answer_entropy.jsonl   # question_id, entropy (token-level)
      splits.json            # train/val/test question_id lists (70/15/15)
      steering/              # per-direction x per-alpha generation results
  features/
    sycophancy{_gptoss,_qwen_moe}/
      L{layer}_K{pct}.npz   # X: (n_samples, hidden_dim), qids: question_ids
results/
  exp_geometry{_gptoss,_qwen_moe}.json
  exp_subspace_analysis{_gptoss,_qwen_moe}.json
  exp_steering{_gptoss}.json
  exp3_probe_by_uncertainty_{entropy_rank,*}.json
  exp4_trajectories{_gptoss,_qwen_moe}.json
  exp5_multiseed{_gptoss,_qwen_moe}.json
  exp_moe_routing{_qwen_moe}.json
  ...
```

## Loading features

```python
import numpy as np

# Load best-layer features for Qwen3-14B
npz = np.load("data/features/sycophancy/L30_K100.npz")
X = npz["X"]       # shape: (5096, 5120) — activations
qids = npz["qids"] # shape: (5096,) — question IDs
```

## Loading labels

```python
import json

with open("data/generated/sycophancy/labeled.jsonl") as f:
    records = [json.loads(line) for line in f]

# Each record has: question_id, question_raw, answer_raw, label (1=sycophantic),
# answer_matching_behavior, thinking (CoT text), uncertainty_score
```

## Key findings

1. **Universal nested geometry**: uncertain sycophancy (w_unc) is a base vector; confident sycophancy adds orthogonal v_override. Holds across all 3 architectures.
2. **Standing committees**: MoE models route both regimes through shared expert subsets (Jaccard >= 0.86).
3. **Steering null result**: Static linear steering produces +/- 2pp change, explained by representational independence.

## Citation

```bibtex
@inproceedings{anonymous2026nested,
  title={The Nested Geometry of Epistemic Capitulation: Asymmetric Transfer of Sycophancy Mechanisms Across {LLM} Architectures},
  author={Anonymous},
  booktitle={ICML 2026 Workshop on Evaluating and Improving Models of Language (EIML)},
  year={2026}
}
```
