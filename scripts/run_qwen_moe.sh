#!/bin/bash
# Full pipeline for Qwen3-30B-A3B (same-family MoE control).
# Prerequisites: generation + labeling already done (labeled.jsonl exists).
#
# Usage: bash scripts/run_qwen_moe.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Extract activations (GPU) ==="
.venv/bin/python -u scripts/extract_qwen_moe.py --device cuda

echo ""
echo "=== 2. Pre-thinking K=0 extraction (GPU) ==="
.venv/bin/python -u scripts/extract_pre_thinking.py --device cuda --model-family qwen_moe

echo ""
echo "=== 3. Precompute features (CPU) ==="
.venv/bin/python -u scripts/precompute_features_qwen_moe.py --n-workers 16

echo ""
echo "=== 4. Answer entropy (GPU) ==="
.venv/bin/python -u scripts/compute_answer_entropy.py --device cuda --model-family qwen_moe

echo ""
echo "=== 5. Exp 3: probes by uncertainty (entropy, rank) ==="
.venv/bin/python -u scripts/exp3_probe_by_uncertainty.py --model-family qwen_moe --uncertainty-source entropy --stratify rank

echo ""
echo "=== 6. Exp 4: trajectories ==="
.venv/bin/python -u scripts/exp4_trajectories.py --model-family qwen_moe

echo ""
echo "=== 7. Exp 5: cross-prediction ==="
.venv/bin/python -u scripts/exp5_cross_prediction.py --model-family qwen_moe

echo ""
echo "=== 8. Exp 5: multi-seed cross-prediction ==="
.venv/bin/python -u scripts/exp5_multiseed.py --model-family qwen_moe

echo ""
echo "=== ALL DONE ==="
