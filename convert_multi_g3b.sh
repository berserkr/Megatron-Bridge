#!/bin/bash
#SBATCH --partition=hpc-mid
#SBATCH --nodes=2
#SBATCH --job-name=granite3b-export
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=144
#SBATCH --output="/mnt/vast/proj/checkpoints/bathen/logs/export-out.%j.log"
#SBATCH --error="/mnt/vast/proj/checkpoints/bathen/logs/export-err.%j.log"
#SBATCH --wait-all-nodes=1
#SBATCH --mem=0

. ~/.bashrc
source ~/run.env

export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp
export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1
export NCCL_COLLNET_ENABLE=0
export NVIDIA_IMEX_CHANNELS=0
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST-}" | head -n1)"
export MASTER_PORT=28444
export NNODES=$SLURM_NNODES

LOCAL_HF_CKPT=/mnt/vast/proj/checkpoints/bathen/models/base/granite-4.1-3b-base-special
SAVED_CKPT=/mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite3b_sft_128k/iter_0002000
EXPORTED_CKPT=/mnt/vast/proj/checkpoints/bathen/models/exports/granite3b_sft_128k

container_mounts="/mnt:/mnt"
container_image="/mnt/vast/squash/nemo_sft_python312_v4.sqsh"

SRUN_ARGS="--kill-on-bad-exit=1 \
            --container-image=${container_image} \
            --container-mounts=${container_mounts} \
            --no-container-remap-root \
            --container-workdir=/mnt/home/bathen/src/github.com/Megatron-Bridge"

export DISTRIBUTED_ARGS=" \
    --nnodes ${NNODES} \
    --nproc_per_node ${GPUS_PER_NODE} \
    --node_rank \$SLURM_NODEID \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}"

# TP=2, PP=1 matches training config (train_granite_3b_128k_8n.yaml)
# --not-strict required for Granite (tied embeddings)
CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun ${DISTRIBUTED_ARGS} \
    examples/conversion/convert_checkpoints_multi_gpu.py export \
    --hf-model ${LOCAL_HF_CKPT} \
    --megatron-path ${SAVED_CKPT} \
    --hf-path ${EXPORTED_CKPT} \
    --tp 2 --pp 1 \
    --not-strict"

echo "$(date) Starting export: ${SAVED_CKPT} -> ${EXPORTED_CKPT}"
srun ${SRUN_ARGS} bash -c "${CMD}"
echo "rc=$?"
