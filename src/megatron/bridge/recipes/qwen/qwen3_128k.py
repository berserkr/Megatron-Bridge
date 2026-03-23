"""
qwen3_8b_full_sft_config.py
---------------------------------
Full SFT training run for Qwen3-8B at 128k sequence length.
Parallelism: TP=4, CP=1, PP=1.
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.training.config import ConfigContainer

# -- The Forge Deployment -----------------------------------------------------
TOTAL_NODES           = 16   # 16 nodes = 64 GPUs. 
CONTEXT_PARALLEL_SIZE = 1

TENSOR_PARALLEL_SIZE   = 4
PIPELINE_PARALLEL_SIZE = 1
SEQ_LENGTH             = 131072

def qwen3_8b_full_sft_config(**kwargs) -> ConfigContainer:
    """Full SFT configuration for Qwen3-8B at 128k."""
    # Pass YAML kwargs (like finetune_lr) into the common builder
    cfg = _sft_common(**kwargs)

    cfg.model = AutoBridge.from_hf_pretrained("/mnt/vast/proj/checkpoints/bathen/models/base/Qwen3-8B-Base").to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = "/mnt/vast/proj/checkpoints/bathen/models/base/Qwen3-8B-Base"

    # -- Parallelism --
    cfg.model.tensor_model_parallel_size           = TENSOR_PARALLEL_SIZE
    cfg.model.pipeline_model_parallel_size         = PIPELINE_PARALLEL_SIZE
    cfg.model.context_parallel_size                = CONTEXT_PARALLEL_SIZE
    cfg.model.sequence_parallel                    = True 
    cfg.model.calculate_per_token_loss             = True
    cfg.model.seq_length                           = SEQ_LENGTH

    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # -- Recompute (Memory Shields) --
    cfg.model.recompute_granularity              = "selective"
    cfg.model.recompute_modules                  = None

    # -- Optimizer precision (Full SFT standard) --
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype              = torch.float32
    cfg.optimizer.main_params_dtype             = torch.float32
    cfg.optimizer.exp_avg_dtype                 = torch.float32
    cfg.optimizer.exp_avg_sq_dtype              = torch.float32

    # -- DDP: Restored for Maximum Speed --
    cfg.ddp.grad_reduce_in_fp32       = True
    cfg.ddp.overlap_grad_reduce       = True    # <-- ON for speed
    cfg.ddp.overlap_param_gather      = True    # <-- ON for speed
    cfg.ddp.check_for_nan_in_grad     = False   # <-- OFF for speed
    cfg.ddp.use_distributed_optimizer = True    # <-- ON to shard optimizer states (ZeRO-1)

    return cfg