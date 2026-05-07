#!/bin/bash
set -euo pipefail
cd /home/viliana-dev/my_projects/sycophancy-uncertainty

echo "=== Geometry for Qwen MoE (CPU only) ==="
.venv/bin/python -u scripts/exp_geometry.py --model-family qwen_moe
echo "Geometry done!"

echo ""
echo "=== MoE Routing for Qwen3-30B-A3B (GPU) ==="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/exp_moe_routing_qwen_moe.py
echo "Routing done!"

echo ""
echo "=== ALL DONE ==="
