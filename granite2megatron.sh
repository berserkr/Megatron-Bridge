  torchrun --nproc_per_node=1 --nnodes=1 examples/conversion/convert_checkpoints.py import \
      --hf-model /mnt/vast/proj/checkpoints/bathen/models/base/30b-soft-lc-512k-lr1e-4-merged-3-7-trans \
      --megatron-path /mnt/vast/proj/checkpoints/bathen/models/base/30b-trans-megatron \
      --torch-dtype bfloat16 \
      --tp 1 --pp 1