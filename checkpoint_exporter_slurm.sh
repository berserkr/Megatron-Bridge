#!/bin/bash
#SBATCH --job-name=dense-export
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=dense_export_%j.log
#SBATCH --error=dense_export_%j.log

set -euo pipefail

# Multi-node Slurm dense Granite export
#
# Usage:
#   # Single node, TP=4
#   sbatch checkpoint_exporter_slurm.sh
#
#   # Override TP via env var
#   TP=2 sbatch checkpoint_exporter_slurm.sh

HF_CONFIG=/mnt/vast/proj/checkpoints/bathen/models/base/granite-4.1-8b-base-special
OUTPUT_BASE=/mnt/vast/proj/checkpoints/bathen/models/sft

TP=${TP:-4}
GPUS_PER_NODE=4
NNODES=${SLURM_NNODES:-1}
NGPUS=${TP}

CHECKPOINTS=(
# /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_run_name/iter_0000553
)

SCRIPT_DIR=/mnt/home/bathen/src/github.com/Megatron-Bridge

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints specified. Edit CHECKPOINTS array in this script."
    exit 1
fi

if [ "${NGPUS}" -gt "${GPUS_PER_NODE}" ] && [ "${NNODES}" -eq 1 ]; then
    echo "WARNING: TP=${NGPUS} > GPUs per node (${GPUS_PER_NODE})."
    echo "         You may need --nodes=$((NGPUS / GPUS_PER_NODE)) or reduce TP."
fi

MASTER_ADDR=$(scontrol show hostname "${SLURM_NODELIST}" | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}

export MASTER_ADDR MASTER_PORT

echo "=== Dense Granite Slurm Export ==="
echo "Job:   ${SLURM_JOB_ID}"
echo "Nodes: ${NNODES} x ${GPUS_PER_NODE} GPUs"
echo "TP=${TP}, total GPUs=${NGPUS}"
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
    output="${OUTPUT_BASE}/${run_name}_${iter_num}"
    echo "=============================="
    echo "Exporting Dense: $ckpt"
    echo "Output:    $output"
    echo "TP=${TP}"
    echo "=============================="

    srun torchrun \
        --nnodes="${NNODES}" \
        --nproc-per-node="${NPROC_PER_NODE}" \
        --rdzv-id="${SLURM_JOB_ID}" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        "${SCRIPT_DIR}/granite_nemo2hf.py" \
            --hf-config "$HF_CONFIG" \
            --megatron-path "$ckpt" \
            --output "$output" \
            --tp "${TP}"

    echo "Done: $iter_name"
    echo ""
done

echo "All dense exports complete."
