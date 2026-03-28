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

"""Granite finetuning recipes with parameterless API.

This module provides pretrain, SFT, and PEFT configurations for IBM Granite models.

Important: Granite models use scaling multipliers (embedding_multiplier,
residual_multiplier, logits_scaling) stored in the HF config. The GraniteBridge
automatically bakes these into the weights during loading. Alternatively, you
can pre-process with transmute_granite.py before training.
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.granite.granite_provider import GraniteModelProvider30B
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.training.config import ConfigContainer

# Pretrained checkpoint path
PRETRAINED_CHECKPOINT = "/mnt/vast/proj/checkpoints/bathen/models/base/30b-soft-lc-512k-lr1e-4-merged-3-7"

# =============================================================================
# Granite 3B (3.0-3b-a800m) Pretrain Configuration
# =============================================================================
def granite_3b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Granite 3B (3.0-3b-a800m).

    Recommended parallelism: TP=1, PP=1 (fits on a single GPU).
    """
    cfg = _pretrain_common()

    # Model config
    hf_path = "ibm-granite/granite-3.0-3b-a800m-base"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Dataset config - mock data by default
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # Model config
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.01

    # Training config
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # DDP config
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


# =============================================================================
# Granite 8B (3.0-8b) Pretrain Configuration
# =============================================================================
def granite_8b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Granite 8B (3.0-8b).

    Recommended parallelism: TP=2, PP=1.
    """
    cfg = _pretrain_common()

    # Model config
    hf_path = "ibm-granite/granite-3.0-8b-base"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Dataset config - mock data by default
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # Model config
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.01

    # Training config
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # DDP config
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


# =============================================================================
# Granite 3B (3.0-3b-a800m) SFT Configuration
# =============================================================================
def granite_3b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Granite 3B.

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs)
    """
    cfg = _sft_common()

    # Model config
    hf_path = "ibm-granite/granite-3.0-3b-a800m-base"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # Global batch size is 8 for packed sequences, 128 otherwise
    cfg.train.global_batch_size = 8
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Training config
    cfg.validation.eval_interval = 30
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 50

    # DDP config
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg


# =============================================================================
# Granite 8B (3.0-8b) SFT Configuration
# =============================================================================
def granite_8b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Granite 8B.

    Recommended parallelism: TP=4, PP=1 (1 node, 8 GPUs)
    """
    cfg = _sft_common()

    # Model config
    hf_path = "ibm-granite/granite-3.0-8b-base"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=True)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # Global batch size is 8 for packed sequences, 128 otherwise
    cfg.train.global_batch_size = 8
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Training config
    cfg.validation.eval_interval = 30
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 50

    # DDP config
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg


# =============================================================================
# Granite 8B Code (3.0-8b-instruct) SFT Configuration
# =============================================================================
def granite_8b_code_sft_config() -> ConfigContainer:
    """Return a full SFT config for Granite 8B Code/Instruct.

    Recommended parallelism: TP=4, PP=1 (1 node, 8 GPUs)
    """
    cfg = _sft_common()

    # Model config
    hf_path = "ibm-granite/granite-3.0-8b-instruct"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # Global batch size is 8 for packed sequences, 128 otherwise
    cfg.train.global_batch_size = 8
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Training config
    cfg.validation.eval_interval = 30
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 50

    # DDP config
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg


# =============================================================================
# PEFT (Parameter-Efficient Fine-Tuning) Configs
# =============================================================================


def granite_3b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Granite 3B.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs)
    """
    cfg = _peft_common()

    # Model config
    hf_path = "ibm-granite/granite-3.0-3b-a800m-base"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Parallelism settings - TP=1 for PEFT (only training adapters)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    cfg.peft = default_peft_config(peft_scheme)

    # Global batch size is 8 for packed sequences, 128 otherwise
    cfg.train.global_batch_size = 8
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Training config
    cfg.validation.eval_interval = 30
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 50

    # DDP config
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg


def granite_8b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Granite 8B.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs)
    """
    cfg = _peft_common()

    # Model config
    hf_path = "ibm-granite/granite-3.0-8b-base"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Parallelism settings - TP=1 for PEFT (only training adapters)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    cfg.peft = default_peft_config(peft_scheme)

    # Global batch size is 8 for packed sequences, 128 otherwise
    cfg.train.global_batch_size = 8
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Training config
    cfg.validation.eval_interval = 30
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 50

    # DDP config
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg


def granite_8b_code_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Granite 8B Code/Instruct.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs)
    """
    cfg = _peft_common()

    # Model config
    hf_path = "ibm-granite/granite-3.0-8b-instruct"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Parallelism settings - TP=1 for PEFT (only training adapters)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    cfg.peft = default_peft_config(peft_scheme)

    # Global batch size is 8 for packed sequences, 128 otherwise
    cfg.train.global_batch_size = 8
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Training config
    cfg.validation.eval_interval = 30
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 50

    # DDP config
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg


# =============================================================================
# Granite 30B Pretrain Configuration
# =============================================================================
def granite_30b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Granite 30B.

    Recommended parallelism: TP=8, PP=2 with recompute enabled for memory optimization.

    Note: Uses GraniteModelProvider30B directly since the model is not yet on HF Hub.
    Update hf_path once the model is published.
    """
    cfg = _pretrain_common()

    # Model config — not on HF yet, use provider directly
    cfg.model = GraniteModelProvider30B()

    # Tokenizer — update path once model is published on HF
    # cfg.tokenizer.tokenizer_model = "ibm-granite/granite-30b-base"

    # Dataset config - mock data by default
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # Model config
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16  # Required for PP > 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.1

    # Training config
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving - ENABLED for 30B (64 layers)
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # DDP config
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


# =============================================================================
# Granite 30B SFT Configuration
# =============================================================================
def granite_30b_sft_config(
    tensor_model_parallel_size: int = 4,
    pipeline_model_parallel_size: int = 8,
    context_parallelism: int = 1,
    sequence_parallelism: bool = True,
    seq_length: int = 131072,
    micro_batch_size: int = 1,
    global_batch_size: int = 32,
    finetune_lr: float = 5e-6,
    lr_warmup_iters: int = 50,
    train_iters: int = 1000,
    packed_sequence: bool = True,
    **kwargs,
) -> ConfigContainer:
    """Granite 30B packed SFT configuration.

    Args:
        tensor_model_parallel_size: TP degree (default 4 = one GB200 node)
        pipeline_model_parallel_size: PP degree (default 8 -> 8 layers/stage)
        context_parallelism: CP degree (1 for 128k, 2+ for longer contexts)
        sequence_parallelism: SP (should be True, required when CP > 1)
        seq_length: Training sequence length
        micro_batch_size: Microbatch size (must be 1 for packed sequences)
        global_batch_size: Global batch size across all DP replicas
        finetune_lr: Peak learning rate
        lr_warmup_iters: LR warmup iterations
        train_iters: Total training iterations
        packed_sequence: Whether to use sequence packing (should be True for SFT)

    Returns:
        ConfigContainer ready for finetune()
    """
    cfg = _sft_common()

    # -- Model ----------------------------------------------------------------
    cfg.model = GraniteModelProvider30B(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        seq_length=seq_length,
    )

    # attention_multiplier is handled via softmax_scale in provider_bridge(),
    # but when constructing the provider directly (not via AutoBridge), we need
    # to read it from the HF config at load time. The bridge's provider_bridge()
    # sets this automatically. For direct construction, the default GPTModel
    # attention uses 1/sqrt(head_dim) which may differ from Granite's
    # attention_multiplier. This is handled at checkpoint load time by the bridge.

    # -- Sequence length (must match on model and dataset) --------------------
    cfg.model.seq_length = seq_length

    # -- Packed sequence specs ------------------------------------------------
    cfg.dataset.packed_sequence_specs.packed_sequence_size = seq_length
    if context_parallelism > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = context_parallelism * 2

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

    # -- Tokenizer (use pretrained checkpoint path) ---------------------------
    cfg.tokenizer.tokenizer_model = PRETRAINED_CHECKPOINT

    # -- Transformer Engine ---------------------------------------------------
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # -- Recompute: full for 128k/CP=1, selective for CP>1 --------------------
    if context_parallelism > 1:
        cfg.model.recompute_granularity = "selective"
    else:
        cfg.model.recompute_granularity = "full"

    # -- Precision ------------------------------------------------------------
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # -- Checkpoint -----------------------------------------------------------
    cfg.checkpoint.pretrained_checkpoint = PRETRAINED_CHECKPOINT
    cfg.checkpoint.save_interval = 100

    # -- DDP ------------------------------------------------------------------
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.check_for_nan_in_grad = True

    # -- Logging --------------------------------------------------------------
    cfg.logger.log_interval = 1
    cfg.validation.eval_interval = 100

    return cfg


# =============================================================================
# Granite 30B PEFT Configuration
# =============================================================================
def granite_30b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Granite 30B.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs).
    Includes recompute for memory optimization.

    Note: Uses GraniteModelProvider30B directly since the model is not yet on HF Hub.
    Update hf_path once the model is published.
    """
    cfg = _peft_common()

    # Model config — not on HF yet, use provider directly
    cfg.model = GraniteModelProvider30B()

    # Tokenizer — update path once model is published on HF
    # cfg.tokenizer.tokenizer_model = "ibm-granite/granite-30b-base"

    # Parallelism settings - TP=1 for PEFT (only training adapters)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    cfg.peft = default_peft_config(peft_scheme)

    # Global batch size is 8 for packed sequences, 128 otherwise
    cfg.train.global_batch_size = 8
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Training config
    cfg.validation.eval_interval = 30
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving - enable recompute for 30B
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 50

    # DDP config
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg
