#!/bin/bash
# Run full gpt-oss experiment pipeline after extraction is done.
# Usage: bash scripts/run_gptoss_experiments.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Precompute features ==="
.venv/bin/python -u scripts/precompute_features_gptoss.py --n-workers 16

echo ""
echo "=== 2. Pre-thinking K=0 extraction ==="
.venv/bin/python -u scripts/extract_pre_thinking.py --device cuda --model-family gptoss

echo ""
echo "=== 3. Answer entropy ==="
.venv/bin/python -u scripts/compute_answer_entropy.py --model openai/gpt-oss-20b --device cuda

echo ""
echo "=== 4. Exp 2: continuous uncertainty ==="
.venv/bin/python -u scripts/exp2_continuous_uncertainty.py --model-family gptoss || echo "exp2 skipped (may need adaptation)"

echo ""
echo "=== 5. Exp 3: probes by uncertainty (heuristic) ==="
.venv/bin/python -u scripts/exp3_probe_by_uncertainty.py --model-family gptoss || echo "exp3 heuristic skipped"

echo ""
echo "=== 6. Exp 3: probes by uncertainty (entropy, rank) ==="
.venv/bin/python -u scripts/exp3_probe_by_uncertainty.py --model-family gptoss --uncertainty-source entropy --stratify rank || echo "exp3 entropy skipped"

echo ""
echo "=== 7. Exp 4: trajectories ==="
.venv/bin/python -u scripts/exp4_trajectories.py --model-family gptoss || echo "exp4 skipped"

echo ""
echo "=== 8. Exp 5: cross-prediction ==="
.venv/bin/python -u scripts/exp5_cross_prediction.py --model-family gptoss || echo "exp5 skipped"

echo ""
echo "=== ALL DONE ==="
