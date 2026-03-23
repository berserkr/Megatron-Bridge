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

# -- Parallelism: SET FOR CP=1 VALIDATION ------------------------------------
TOTAL_NODES           = 16   
CONTEXT_PARALLEL_SIZE = 1    # Setting to 1 to test baseline/fixes

# -- Derived ------------------------------------------------------------------
TENSOR_PARALLEL_SIZE   = 8
PIPELINE_PARALLEL_SIZE = 1
GPUS_PER_NODE           = 4

def qwen3_8b_cp_validation_config() -> ConfigContainer:
    cfg = _sft_common()

    # -- Model & Tokenizer (Keep your existing paths) ------------------------
    cfg.model = AutoBridge.from_hf_pretrained("/mnt/vast/proj/checkpoints/bathen/models/base/Qwen3-8B-Base").to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = "/mnt/vast/proj/checkpoints/bathen/models/base/Qwen3-8B-Base"

    # -- Parallelism ----------------------------------------------------------
    cfg.model.tensor_model_parallel_size           = TENSOR_PARALLEL_SIZE
    cfg.model.pipeline_model_parallel_size         = PIPELINE_PARALLEL_SIZE
    cfg.model.context_parallel_size                = CONTEXT_PARALLEL_SIZE
    cfg.model.sequence_parallel                    = True # Keep True to shard activations across TP=4
    cfg.model.calculate_per_token_loss             = True

    # -- Sequence length ------------------------------------------------------
    cfg.model.seq_length = 131072

    # -- Training Specs -------------------------------------------------------
    cfg.train.train_iters       = 20
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size  = 1

    # -- Transformer Engine & Kernels -----------------------------------------
    cfg.model.transformer_impl = "transformer_engine"
    # Ensure FlashAttention is forced for the long context
    cfg.model.attention_backend = "flash_attn" 

    # -- RECOMPUTE: THE CRITICAL CHANGE FOR CP=1 ------------------------------
    # 'selective' will OOM at 128k/CP=1. 'full' is required.
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method      = "block" # Recompute at the layer level
    cfg.model.recompute_num_layers  = 1       # Recompute every layer

    # -- OPTIMIZER: THE SECOND CRITICAL CHANGE --------------------------------
    # Sharding the optimizer states across DP ranks to save ~40-50GB VRAM
    cfg.ddp.use_distributed_optimizer = True
    
    # -- Precision ------------------------------------------------------------
    # Keeping float32 for master weights as per your original for stability
    cfg.optimizer.main_grads_dtype  = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32

    # -- DDP & Debugging ------------------------------------------------------
    cfg.ddp.overlap_grad_reduce     = False
    cfg.ddp.overlap_param_gather    = False
    cfg.ddp.check_for_nan_in_grad   = True

    return cfg