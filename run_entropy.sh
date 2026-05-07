#!/bin/bash
GPUS=(1 3 4 5 6 7)
NUM_SHARDS=6
mkdir -p logs
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u scripts/compute_answer_entropy.py \
        --device cuda --shard-id $i --num-shards $NUM_SHARDS \
        > logs/entropy_shard_${i}.log 2>&1 &
done
wait
