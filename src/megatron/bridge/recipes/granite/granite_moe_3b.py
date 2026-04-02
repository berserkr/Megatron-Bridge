# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Granite MoE 3B SFT recipe for Megatron-Bridge.

Sparse MoE transformer, 32 layers, hidden=1536, ffn=512 per expert,
24 heads, 8 KV heads, 40 experts, top-8 routing.
~3B total parameters, ~800M active per token.

Uses AutoBridge with load_weights=True to load the original HF checkpoint.
AutoBridge + GraniteMoeBridge handles all Granite multipliers correctly:
  - embedding_multiplier, residual_multiplier, logits_scaling -> baked into weights
  - attention_multiplier -> set as softmax_scale on the provider
  - tie_word_embeddings -> untied, separate scaling for embed and output_layer

Parallelism defaults target 16 nodes x 4 GB200 = 64 GPUs:
  TP=1, PP=1, EP=8, CP=4, SP=False
  -> 32 GPUs/model replica, 2 DP replicas

Set FILELOCK_LOCK_CLASS=soft in launch env to avoid NFS lock issues at scale.
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


def granite_moe_3b_finetune_config(
    hf_model_path: str = "ibm/PowerMoE-3b",
    pretrained_checkpoint: str = None,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 8,
    context_parallelism: int = 4,
    sequence_parallelism: bool = False,
    seq_length: int = 131072,
    micro_batch_size: int = 1,
    global_batch_size: int = 32,
    finetune_lr: float = 5e-6,
    lr_warmup_iters: int = 50,
    train_iters: int = 1000,
    packed_sequence: bool = True,
    **kwargs,
) -> ConfigContainer:
    """Granite MoE 3B packed SFT configuration.

    Args:
        hf_model_path: Path to HF Granite MoE checkpoint
        pretrained_checkpoint: Optional Megatron checkpoint to resume from
        tensor_model_parallel_size: TP degree (default 1, model is small)
        pipeline_model_parallel_size: PP degree (default 1)
        expert_model_parallel_size: EP degree (default 8, distributes 40 experts)
        context_parallelism: CP degree (4 for 128k -> 32k tokens/GPU)
        sequence_parallelism: SP (False when TP=1)
        seq_length: Training sequence length
        micro_batch_size: Microbatch size (must be 1 for packed sequences)
        global_batch_size: Global batch size across all DP replicas
        finetune_lr: Peak learning rate
        lr_warmup_iters: LR warmup iterations
        train_iters: Total training iterations
        packed_sequence: Whether to use sequence packing

    Returns:
        ConfigContainer ready for finetune()
    """
    cfg = _sft_common()

    # -- Model via AutoBridge (handles all Granite multipliers correctly) ------
    cfg.model = AutoBridge.from_hf_pretrained(hf_model_path).to_megatron_provider(load_weights=True)

    # -- Parallelism ----------------------------------------------------------
    cfg.model.tensor_model_parallel_size = tensor_model_parallel_size
    cfg.model.pipeline_model_parallel_size = pipeline_model_parallel_size
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = context_parallelism
    cfg.model.sequence_parallel = sequence_parallelism
    cfg.model.expert_model_parallel_size = expert_model_parallel_size

    # -- Sequence length (must match on model and dataset) --------------------
    cfg.model.seq_length = seq_length

    # -- Packed sequence specs ------------------------------------------------
    cfg.dataset.packed_sequence_specs.packed_sequence_size = seq_length
    if context_parallelism > 1:
        # Correct formula: TP x CP (NOT CP x 2)
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = tensor_model_parallel_size * context_parallelism

    # -- Per-token loss (required for correct packed SFT loss averaging) ------
    cfg.model.calculate_per_token_loss = True

    # -- Training config ------------------------------------------------------
    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = micro_batch_size

    # -- Optimizer: cosine annealing with warmup ------------------------------
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        max_lr=finetune_lr,
        min_lr=0.0,
        lr_warmup_iters=lr_warmup_iters,
        adam_beta2=0.98,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # -- Tokenizer ------------------------------------------------------------
    cfg.tokenizer.tokenizer_model = hf_model_path

    # -- Transformer Engine ---------------------------------------------------
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # -- Attention backend: flash for long context ----------------------------
    cfg.model.attention_backend = "flash_attn"

    # -- MoE-specific settings ------------------------------------------------
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_permute_fusion = True

    # -- Recompute: selective for MoE (save memory on attention) --------------
    cfg.model.recompute_granularity = "selective"

    # -- Precision ------------------------------------------------------------
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32

    # -- Checkpoint: weights loaded via AutoBridge load_weights=True ----------
    if pretrained_checkpoint:
        cfg.checkpoint.pretrained_checkpoint = pretrained_checkpoint
    cfg.checkpoint.save_interval = 250

    # -- DDP ------------------------------------------------------------------
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True

    # -- Logging --------------------------------------------------------------
    cfg.logger.log_interval = 1
    cfg.validation.eval_interval = 100

    print(f"DEBUG softmax_scale on provider: {cfg.model.softmax_scale}")
    return cfg
