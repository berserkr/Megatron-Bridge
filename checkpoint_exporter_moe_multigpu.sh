#!/bin/bash
set -euo pipefail

# Multi-GPU single-node MoE export (4x GB200)
# Set TP and EP to match training parallelism, or let it reshard.
# World size must equal TP * EP.

HF_CONFIG=/mnt/vast/proj/checkpoints/bathen/models/base/powermoe-3b
OUTPUT_BASE=/mnt/vast/proj/checkpoints/bathen/models/sft

TP=${TP:-2}
EP=${EP:-2}
NGPUS=$((TP * EP))

CHECKPOINTS=(
# /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granitemoe_run_name/iter_0001000
)

SCRIPT_DIR=/mnt/home/bathen/src/github.com/Megatron-Bridge

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints specified. Edit CHECKPOINTS array in this script."
    exit 1
fi

echo "=== MoE Multi-GPU Export ==="
echo "TP=${TP}, EP=${EP}, GPUs=${NGPUS}"
echo ""

for ckpt in "${CHECKPOINTS[@]}"; do
    iter_name=$(basename "$ckpt")
    iter_num="${iter_name#iter_}"
    run_name=$(basename "$(dirname "$ckpt")")
    output="${OUTPUT_BASE}/${run_name}_moe_${iter_num}"
    echo "=============================="
    echo "Exporting MoE (${NGPUS} GPUs): $ckpt"
    echo "Output:    $output"
    echo "TP=${TP}, EP=${EP}"
    echo "=============================="
    torchrun --nproc-per-node="${NGPUS}" "${SCRIPT_DIR}/granite_moe_nemo2hf.py" \
        --hf-config "$HF_CONFIG" \
        --megatron-path "$ckpt" \
        --output "$output" \
        --tp "${TP}" \
        --ep "${EP}"
    echo "Done: $iter_name"
    echo ""
done

echo "All MoE exports complete."
