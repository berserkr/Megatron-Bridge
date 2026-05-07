#!/bin/bash
set -euo pipefail

# Multi-GPU single-node dense Granite export (4x GB200)
# Set TP to match training parallelism, or let it reshard.
# World size must equal TP (no EP for dense).

HF_CONFIG=/mnt/vast/proj/checkpoints/bathen/models/base/granite-4.1-8b-base-special
OUTPUT_BASE=/mnt/vast/proj/checkpoints/bathen/models/sft

TP=${TP:-4}
NGPUS=${TP}

CHECKPOINTS=(
# /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_run_name/iter_0000553
)

SCRIPT_DIR=/mnt/home/bathen/src/github.com/Megatron-Bridge

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints specified. Edit CHECKPOINTS array in this script."
    exit 1
fi

echo "=== Dense Granite Multi-GPU Export ==="
echo "TP=${TP}, GPUs=${NGPUS}"
echo ""

for ckpt in "${CHECKPOINTS[@]}"; do
    iter_name=$(basename "$ckpt")
    iter_num="${iter_name#iter_}"
    run_name=$(basename "$(dirname "$ckpt")")
    output="${OUTPUT_BASE}/${run_name}_${iter_num}"
    echo "=============================="
    echo "Exporting Dense (${NGPUS} GPUs): $ckpt"
    echo "Output:    $output"
    echo "TP=${TP}"
    echo "=============================="
    torchrun --nproc-per-node="${NGPUS}" "${SCRIPT_DIR}/granite_nemo2hf.py" \
        --hf-config "$HF_CONFIG" \
        --megatron-path "$ckpt" \
        --output "$output" \
        --tp "${TP}"
    echo "Done: $iter_name"
    echo ""
done

echo "All dense exports complete."
