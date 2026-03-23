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
