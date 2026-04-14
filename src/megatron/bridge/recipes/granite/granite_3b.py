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
Granite 3B Dense SFT recipe for Megatron-Bridge.

Dense transformer (no MoE), 40 layers, hidden=2560, ffn=8192, 40 heads, 8 KV heads.
~3B parameters. vocab_size=100352, rope_theta=10M (128k native context).

Uses AutoBridge with load_weights=True to load the original HF checkpoint.
AutoBridge + GraniteBridge handles all Granite multipliers correctly:
  - embedding_multiplier (12.0), residual_multiplier (0.22), logits_scaling (10.0) -> baked into weights
  - attention_multiplier (0.015625) -> set as softmax_scale on the provider
  - tie_word_embeddings -> untied, separate scaling for embed and output_layer

Note: vocab_size=100352 is divisible by 128 and by common TP sizes (1, 2, 4, 8).
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


def granite_3b_finetune_config(
    hf_model_path: str = "ibm-granite/granite-3.3-3b-instruct",
    pretrained_checkpoint: str = None,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    seq_length: int = 8192,
    micro_batch_size: int = 1,
    global_batch_size: int = 32,
    finetune_lr: float = 5e-6,
    lr_warmup_iters: int = 50,
    train_iters: int = 1000,
    packed_sequence: bool = True,
    **kwargs,
) -> ConfigContainer:
    """Granite 3B Dense packed SFT configuration.

    Args:
        hf_model_path: Path to HF Granite 3B dense checkpoint
        pretrained_checkpoint: Optional Megatron checkpoint to resume from
        tensor_model_parallel_size: TP degree (default 1, model fits on 1 GPU)
        pipeline_model_parallel_size: PP degree (default 1)
        context_parallelism: CP degree (default 1)
        sequence_parallelism: SP (default False when TP=1)
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

    # -- Sequence length (must match on model and dataset) --------------------
    cfg.model.seq_length = seq_length

    # -- Packed sequence specs ------------------------------------------------
    cfg.dataset.packed_sequence_specs.packed_sequence_size = seq_length
    if context_parallelism > 1:
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

    # -- Recompute: selective for CP>1, full block for CP=1 -------------------
    if context_parallelism > 1:
        cfg.model.recompute_granularity = "selective"
    else:
        cfg.model.recompute_granularity = "full"
        cfg.model.recompute_method = "block"
        cfg.model.recompute_num_layers = 1

    # -- Precision ------------------------------------------------------------
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32

    # -- Checkpoint: weights loaded via AutoBridge load_weights=True ----------
    if pretrained_checkpoint:
        cfg.checkpoint.pretrained_checkpoint = pretrained_checkpoint
    cfg.checkpoint.save_interval = 100

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
