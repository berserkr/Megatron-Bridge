#!/bin/bash

torchrun --nproc_per_node 4 convert_checkpoints.py import --hf-model /mnt/vast/proj/checkpoints/bathen/models/base/30b-soft-lc-512k-lr1e-4-merged-3-7 --megatron-path /mnt/vast/proj/checkpoints/bathen/models/base/30b-soft-lc-512k-lr1e-4-merged-3-7-bridge/
