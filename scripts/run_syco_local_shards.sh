#!/bin/bash
# Launch one sycophancy shard on GPU (SHARD + 4).
# Usage: bash scripts/run_syco_local_shards.sh <SHARD>  (SHARD = 0,1,2,3)
# Maps shard 0->GPU4, 1->GPU5, 2->GPU6, 3->GPU7.
# 80GB A800s: batch_size=8 fits easily (model ~28GB + ~10GB KV cache).
SHARD=$1
GPU=$((SHARD + 4))
cd "$(dirname "$0")/.."
mkdir -p logs
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. /home/viliana-dev/.local/bin/uv run python -m src.generate \
  --behavior sycophancy --n-samples 5100 --local --device cuda \
  --skip-gemini --shard-id $SHARD --num-shards 4 --batch-size 8 \
  2>&1 | tee logs/syco_local_s${SHARD}.log
