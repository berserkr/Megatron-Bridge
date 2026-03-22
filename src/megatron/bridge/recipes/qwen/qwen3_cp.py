"""
qwen3_8b_cp_validation_config.py
---------------------------------
Short validation run to prove CP>1 works at 128k on Qwen3-8B (dense transformer,
no Mamba layers). If this passes cleanly, CP>1 is validated at the infrastructure
level before porting to the Super model.

What "passing" looks like:
  - No NaN/Inf in gradients or loss for all N steps
  - Loss decreases or is stable (not exploding)
  - No NCCL timeouts or ring-attention shape errors

Hardware : 16 nodes x 4x GB200 = 64 GPUs  (validation run)
           64 nodes x 4x GB200 = 256 GPUs (production — change TOTAL_NODES below)

Parallelism: TP=4, CP=2, PP=1  ->  8 GPUs/replica, 8 DP replicas (16 nodes)
             Change CONTEXT_PARALLEL_SIZE=4 for the second validation pass.
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.training.config import ConfigContainer

# -- Parallelism: change only these two lines to switch between runs ----------
TOTAL_NODES           = 16   # 16 for validation, 64 for production
CONTEXT_PARALLEL_SIZE = 2    # 2 for first pass, 4 for second pass

# -- Derived (do not change) --------------------------------------------------
TENSOR_PARALLEL_SIZE   = 4
PIPELINE_PARALLEL_SIZE = 1
GPUS_PER_NODE          = 4
TOTAL_GPUS             = TOTAL_NODES * GPUS_PER_NODE


DP         = TOTAL_GPUS // (TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE)
SEQ_LENGTH = 131072


def qwen3_8b_cp_validation_config() -> ConfigContainer:
    """Short CP>1 validation run for Qwen3-8B at 128k.

    Intentionally minimal training steps -- just enough to confirm:
      1. No NaNs in loss or gradients
      2. Ring-attention runs without shape/NCCL errors
      3. Loss is stable across steps

    After validating CP=2, change CONTEXT_PARALLEL_SIZE=4 and re-run.
    """
    cfg = _sft_common()

    # -- Model ----------------------------------------------------------------
    cfg.model = AutoBridge.from_hf_pretrained("/mnt/vast/proj/checkpoints/bathen/models/base/Qwen3-8B-Base").to_megatron_provider(load_weights=False)

    # -- Tokenizer ------------------------------------------------------------
    cfg.tokenizer.tokenizer_model = "/mnt/vast/proj/checkpoints/bathen/models/base/Qwen3-8B-Base"

    # -- Parallelism ----------------------------------------------------------
    cfg.model.tensor_model_parallel_size           = TENSOR_PARALLEL_SIZE
    cfg.model.pipeline_model_parallel_size         = PIPELINE_PARALLEL_SIZE
    cfg.model.pipeline_model_parallel_layout       = None
    cfg.model.pipeline_dtype                       = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size                = CONTEXT_PARALLEL_SIZE
    cfg.model.sequence_parallel                    = True  # required when CP > 1
    cfg.model.calculate_per_token_loss             = True

    # -- Sequence length ------------------------------------------------------
    cfg.model.seq_length = SEQ_LENGTH

    # -- packed_sequence_specs: follow the same pattern as the upstream recipes
    # (set pad_seq_to_mult when CP > 1, matching qwen3_8b_sft_config exactly)
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # -- Validation run: small step count -------------------------------------
    # 20 steps is enough to catch NaNs, shape errors, and unstable loss.
    cfg.train.train_iters       = 20
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size  = 1

    # -- Logging: verbose for debugging ---------------------------------------
    cfg.logger.log_interval      = 1   # loss at every step
    cfg.validation.eval_interval = 10  # eval twice in 20 steps

    # -- Training config ------------------------------------------------------
    cfg.train.manual_gc          = False
    cfg.train.manual_gc_interval = 0

    # -- Transformer Engine ---------------------------------------------------
    cfg.model.transformer_impl = "transformer_engine"

    # -- CUDA Graph: off for validation (can hide errors) ---------------------
    cfg.model.cuda_graph_impl         = "none"
    cfg.model.cuda_graph_scope        = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # -- Kernels --------------------------------------------------------------
    cfg.model.attention_backend         = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # -- Recompute ------------------------------------------------------------
    cfg.model.recompute_granularity              = "selective"
    cfg.model.recompute_modules                  = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules                    = None

    # -- Precision: bf16, no FP8 (isolate CP as the only variable) -----------

    # -- Optimizer precision --------------------------------------------------
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype              = torch.float32
    cfg.optimizer.main_params_dtype             = torch.float32
    cfg.optimizer.exp_avg_dtype                 = torch.float32
    cfg.optimizer.exp_avg_sq_dtype              = torch.float32

    # -- Checkpoint -----------------------------------------------------------
    cfg.checkpoint.save_interval = 20
    # cfg.checkpoint.save                  = "/path/to/cp_validation/save"
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/pretrained"

    # -- DDP: NaN checking ON, overlaps OFF for clean error attribution -------
    cfg.ddp.grad_reduce_in_fp32       = False
    cfg.ddp.overlap_grad_reduce       = False
    cfg.ddp.overlap_param_gather      = False
    cfg.ddp.check_for_nan_in_grad     = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg