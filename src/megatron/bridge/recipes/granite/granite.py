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
def granite_30b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Granite 30B.

    Recommended parallelism: TP=8, PP=2 (2 nodes, 16 GPUs total).
    Includes recompute for memory optimization.

    Note: Uses GraniteModelProvider30B directly since the model is not yet on HF Hub.
    Update hf_path once the model is published.
    """
    cfg = _sft_common()

    # Model config — not on HF yet, use provider directly
    cfg.model = GraniteModelProvider30B()

    # Tokenizer — update path once model is published on HF
    # cfg.tokenizer.tokenizer_model = "ibm-granite/granite-30b-base"

    # Parallelism settings - needs more parallelism for 30B (64 layers)
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16  # Required for PP > 1
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
