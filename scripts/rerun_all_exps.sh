#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Qwen 14B ==="
.venv/bin/python -u scripts/exp3_probe_by_uncertainty.py --uncertainty-source entropy --stratify rank
.venv/bin/python -u scripts/exp4_trajectories.py
.venv/bin/python -u scripts/exp5_cross_prediction.py
.venv/bin/python -u scripts/exp5_multiseed.py

echo "=== gpt-oss ==="
.venv/bin/python -u scripts/exp3_probe_by_uncertainty.py --model-family gptoss --uncertainty-source entropy --stratify rank
.venv/bin/python -u scripts/exp4_trajectories.py --model-family gptoss
.venv/bin/python -u scripts/exp5_cross_prediction.py --model-family gptoss
.venv/bin/python -u scripts/exp5_multiseed.py --model-family gptoss

echo "=== Qwen MoE ==="
.venv/bin/python -u scripts/exp3_probe_by_uncertainty.py --model-family qwen_moe --uncertainty-source entropy --stratify rank
.venv/bin/python -u scripts/exp4_trajectories.py --model-family qwen_moe
.venv/bin/python -u scripts/exp5_cross_prediction.py --model-family qwen_moe
.venv/bin/python -u scripts/exp5_multiseed.py --model-family qwen_moe

echo "=== ALL DONE ==="
