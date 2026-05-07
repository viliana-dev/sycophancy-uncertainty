#!/bin/bash
cd ~/my_projects/sycophancy-uncertainty

GPUS=(1 3 4 5 6 7)
NUM_SHARDS=6

for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    SHARD=$i
    echo "Starting shard $SHARD on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -m src.extract \
        --device cuda --shard-id $SHARD --num-shards $NUM_SHARDS \
        > logs/extract_shard_${SHARD}.log 2>&1 &
done

echo "All $NUM_SHARDS shards launched. Logs in logs/"
echo "Monitor: tail -f logs/extract_shard_*.log"
wait
echo "All shards complete!"
