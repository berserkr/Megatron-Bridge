# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from transformers.activations import ACT2FN
from megatron.bridge.models.gpt_provider import GPTModelProvider

logger = logging.getLogger(__name__)

# Granite uses SiLU (Swish) for its SwiGLU activation
def silu_activation(x):
    return torch.nn.functional.silu(x)

@dataclass
class GraniteModelProvider(GPTModelProvider):
    """Configuration class for Granite models in Megatron."""

    # Normalization & Activations
    normalization: str = "RMSNorm"
    activation_func: Callable = silu_activation
    position_embedding_type: str = "rope"
    gated_linear_unit: bool = True  # Crucial for Granite's gate_proj + up_proj
    add_bias_linear: bool = False
    
    # Fusions & Dropout
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False # Must be false to split multipliers
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_add_fusion: bool = False
    apply_rope_fusion: bool = True

    # Data type settings to match HF models
    bf16: bool = True
    fp16: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16

@dataclass
class GraniteModelProvider230B(GraniteModelProvider):
    """
    Configuration class for the 230B+ Granite model.
    Verify these dimensions against your Hugging Face config.json.
    """
    num_layers: int = 80
    seq_length: int = 4096
    hidden_size: int = 16384
    ffn_hidden_size: int = 49152  # Typically 3x hidden_size for SwiGLU
    num_attention_heads: int = 128
    num_query_groups: Optional[int] = 8  # For Grouped Query Attention
    kv_channels: Optional[int] = 128
    init_method_std: float = 0.005
    