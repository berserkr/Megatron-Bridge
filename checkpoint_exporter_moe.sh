#!/bin/bash
set -euo pipefail

# Single-GPU MoE export — for quick testing (TP=1, EP=1)
# Reshards from any training parallelism internally.

HF_CONFIG=/mnt/vast/proj/checkpoints/bathen/models/base/powermoe-3b
OUTPUT_BASE=/mnt/vast/proj/checkpoints/bathen/models/sft

CHECKPOINTS=(
# /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granitemoe_run_name/iter_0001000
)

SCRIPT_DIR=/mnt/home/bathen/src/github.com/Megatron-Bridge

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints specified. Edit CHECKPOINTS array in this script."
    exit 1
fi

for ckpt in "${CHECKPOINTS[@]}"; do
    iter_name=$(basename "$ckpt")              # iter_0001000
    iter_num="${iter_name#iter_}"              # 0001000
    run_name=$(basename "$(dirname "$ckpt")")  # granitemoe_run_name
    output="${OUTPUT_BASE}/${run_name}_moe_${iter_num}"
    echo "=============================="
    echo "Exporting MoE (single GPU): $ckpt"
    echo "Output:    $output"
    echo "=============================="
    torchrun --nproc-per-node=1 "${SCRIPT_DIR}/granite_moe_nemo2hf.py" \
        --hf-config "$HF_CONFIG" \
        --megatron-path "$ckpt" \
        --output "$output" \
        --ep 1
    echo "Done: $iter_name"
    echo ""
done

echo "All MoE exports complete."
