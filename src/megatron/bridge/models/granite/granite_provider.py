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
Granite Model Provider configurations for Megatron-Core.

Granite models (ibm-granite family) require special handling during weight loading
because the HuggingFace checkpoint stores scaling multipliers in the config
(embedding_multiplier, residual_multiplier, logits_scaling) rather than baking
them into the weights. The GraniteBridge handles this transparently.

If you pre-process weights with transmute_granite.py (which bakes the multipliers
into the weights and resets the config values to 1.0), the bridge will detect this
and skip the scaling step automatically.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider


logger = logging.getLogger(__name__)


@dataclass
class GraniteModelProvider(GPTModelProvider):
    """
    Base configuration class for Granite models in Megatron.

    Granite (ibm-granite) uses a standard decoder-only transformer with:
      - RMSNorm normalization
      - RoPE positional embeddings
      - SwiGLU activation (gated MLP)
      - Grouped Query Attention (GQA)
      - Scaling multipliers in the HF config (handled by the bridge during loading)
    """

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    position_embedding_type: str = "rope"
    gated_linear_unit: bool = True
    add_bias_linear: bool = False

    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False

    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_add_fusion: bool = False
    apply_rope_fusion: bool = True

    params_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True
    fp16: bool = False


@dataclass
class GraniteModelProvider3B(GraniteModelProvider):
    """
    Config for Granite 3.0 3B-A800M: https://huggingface.co/ibm-granite/granite-3.0-3b-a800m-base

    MoE variant with 800M active parameters out of 3B total.
    """

    num_layers: int = 32
    hidden_size: int = 3072
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 24
    num_query_groups: int = 8
    seq_length: int = 4096
    rotary_base: float = 500000.0
    init_method_std: float = 0.01
    layernorm_epsilon: float = 1e-5
    vocab_size: int = 49152


@dataclass
class GraniteModelProvider8B(GraniteModelProvider):
    """
    Config for Granite 3.0 8B: https://huggingface.co/ibm-granite/granite-3.0-8b-base
    """

    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    num_query_groups: int = 8
    seq_length: int = 4096
    rotary_base: float = 500000.0
    init_method_std: float = 0.01
    layernorm_epsilon: float = 1e-5
    vocab_size: int = 49152


@dataclass
class GraniteModelProvider8BCode(GraniteModelProvider):
    """
    Config for Granite 3.0 8B Code: https://huggingface.co/ibm-granite/granite-3.0-8b-instruct
    """

    num_layers: int = 36
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    num_query_groups: int = 8
    seq_length: int = 4096
    rotary_base: float = 500000.0
    init_method_std: float = 0.01
    layernorm_epsilon: float = 1e-5
    vocab_size: int = 49152


@dataclass
class GraniteModelProvider30B(GraniteModelProvider):
    """
    Config for Granite 30B.
    """

    num_layers: int = 64
    hidden_size: int = 4096
    ffn_hidden_size: int = 32768
    num_attention_heads: int = 32
    num_query_groups: int = 8
    seq_length: int = 131072
    rotary_base: float = 50000000.0
    init_method_std: float = 0.1
    layernorm_epsilon: float = 1e-5
    vocab_size: int = 100352


@dataclass
class GraniteMoeModelProvider(GraniteModelProvider):
    """
    Base configuration for Granite MoE models in Megatron.

    Granite MoE uses the same architecture as dense Granite (RMSNorm, RoPE,
    SwiGLU, GQA, scaling multipliers) but replaces the dense MLP with a
    Sparse Mixture-of-Experts layer using top-k gating.

    The HF checkpoint stores expert weights in grouped format:
      - input_linear.weight: [num_experts, intermediate_size*2, hidden_size] (fused gate+up)
      - output_linear.weight: [num_experts, hidden_size, intermediate_size] (down proj)
    """

    # MoE architecture
    num_moe_experts: int = 40
    moe_router_topk: int = 8
    moe_ffn_hidden_size: int = 512

    # MoE training
    moe_aux_loss_coeff: float = 0.001
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_router_pre_softmax: bool = False
    moe_grouped_gemm: bool = True
    moe_router_score_function: str = "softmax"
    moe_permute_fusion: bool = True
    moe_router_dtype: str = "fp32"