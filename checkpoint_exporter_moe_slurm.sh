#!/bin/bash
#SBATCH --job-name=moe-export
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=moe_export_%j.log
#SBATCH --error=moe_export_%j.log

set -euo pipefail

# Multi-node Slurm MoE export
# Adjust --nodes above if TP * EP > GPUs per node (4x GB200).
#
# Usage:
#   # Single node, TP=2 EP=2
#   sbatch checkpoint_exporter_moe_slurm.sh
#
#   # Override parallelism via env vars
#   TP=4 EP=1 sbatch checkpoint_exporter_moe_slurm.sh
#
#   # Multi-node (edit --nodes above to match)
#   TP=4 EP=2 sbatch --nodes=2 checkpoint_exporter_moe_slurm.sh

HF_CONFIG=/mnt/vast/proj/checkpoints/bathen/models/base/powermoe-3b
OUTPUT_BASE=/mnt/vast/proj/checkpoints/bathen/models/sft

TP=${TP:-2}
EP=${EP:-2}
GPUS_PER_NODE=4
NNODES=${SLURM_NNODES:-1}
NGPUS=$((TP * EP))

CHECKPOINTS=(
# /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granitemoe_run_name/iter_0001000
)

SCRIPT_DIR=/mnt/home/bathen/src/github.com/Megatron-Bridge

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints specified. Edit CHECKPOINTS array in this script."
    exit 1
fi

if [ "${NGPUS}" -gt "${GPUS_PER_NODE}" ] && [ "${NNODES}" -eq 1 ]; then
    echo "WARNING: TP*EP=${NGPUS} > GPUs per node (${GPUS_PER_NODE})."
    echo "         You may need --nodes=$((NGPUS / GPUS_PER_NODE)) or reduce TP/EP."
fi

# Resolve master address for multi-node
MASTER_ADDR=$(scontrol show hostname "${SLURM_NODELIST}" | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}

export MASTER_ADDR MASTER_PORT

echo "=== MoE Slurm Export ==="
echo "Job:   ${SLURM_JOB_ID}"
echo "Nodes: ${NNODES} x ${GPUS_PER_NODE} GPUs"
echo "TP=${TP}, EP=${EP}, total GPUs=${NGPUS}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo ""

NPROC_PER_NODE=${GPUS_PER_NODE}
if [ "${NGPUS}" -le "${GPUS_PER_NODE}" ]; then
    NPROC_PER_NODE=${NGPUS}
fi

for ckpt in "${CHECKPOINTS[@]}"; do
    iter_name=$(basename "$ckpt")
    iter_num="${iter_name#iter_}"
    run_name=$(basename "$(dirname "$ckpt")")
    output="${OUTPUT_BASE}/${run_name}_moe_${iter_num}"
    echo "=============================="
    echo "Exporting MoE: $ckpt"
    echo "Output:    $output"
    echo "TP=${TP}, EP=${EP}"
    echo "=============================="

    srun torchrun \
        --nnodes="${NNODES}" \
        --nproc-per-node="${NPROC_PER_NODE}" \
        --rdzv-id="${SLURM_JOB_ID}" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        "${SCRIPT_DIR}/granite_moe_nemo2hf.py" \
            --hf-config "$HF_CONFIG" \
            --megatron-path "$ckpt" \
            --output "$output" \
            --tp "${TP}" \
            --ep "${EP}"

    echo "Done: $iter_name"
    echo ""
done

echo "All MoE exports complete."
