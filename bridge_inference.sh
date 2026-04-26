#!/bin/bash

torchrun --nproc-per-node=4 examples/conversion/hf_to_megatron_generate_text.py \
      --hf_model_path /mnt/vast/proj/checkpoints/bathen/models/base/granite-4.1-8b-base-special \
      --megatron_model_path /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_42_8b_sft_32k_cp2_1ep_stage2/iter_0000540 \
      --tp 4 \
      --prompt "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nA 2 kg block slides down a frictionless 30-degree incline. Calculate the acceleration and the speed after traveling 5 meters from rest.<|im_end|>\n<|im_start|>assistant\n<think>\n" \
      --max_new_tokens 512

