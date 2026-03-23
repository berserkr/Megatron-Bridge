#!/bin/bash

LOCAL_HF_CKPT=/mnt/vast/proj/checkpoints/bathen/models/base/Qwen3-8B-Base
SAVED_CKPT=/mnt/vast/proj/checkpoints/bathen/models/sft/Qwen3-8B-Base-cp4-256k-test-16nodes/iter_0000100
EXPORTED_CKPT=/mnt/vast/proj/checkpoints/bathen/models/exports/Qwen3-8B-Base-cp4-256k-test-16nodes-hf
TP=4
PP=1
EP=1
ETP=1

torchrun --nproc_per_node 4 examples/conversion/convert_checkpoints.py export \
    --hf-model $LOCAL_HF_CKPT \
    --megatron-path $SAVED_CKPT \
    --hf-path $EXPORTED_CKPT \
    --tp $TP --pp $PP --ep $EP --etp $ETP
