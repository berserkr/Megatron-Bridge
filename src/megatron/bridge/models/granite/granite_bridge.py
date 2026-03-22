# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import GraniteForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    QKVMapping,
    GatedMLPMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

from .granite_provider import GraniteModelProvider


@MegatronModelBridge.register_bridge(
    source=GraniteForCausalLM,
    target=GPTModel,
    provider=GraniteModelProvider,
    model_type="granite",
)
class GraniteBridge(MegatronModelBridge):
    """
    Megatron Bridge for Granite Causal LM.
    """

    CONFIG_MAPPING = MegatronModelBridge.CONFIG_MAPPING + [
        ("rms_norm_eps", "layernorm_epsilon"),
    ]

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GraniteModelProvider:
        """Convert HuggingFace Granite config to GraniteModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        config = hf_pretrained.config

        provider.normalization = "RMSNorm"
        provider.position_embedding_type = "rope"
        provider.gated_linear_unit = True
        provider.add_bias_linear = getattr(config, "attention_bias", False)
        
        # We enforce untied weights here so the Alchemist's baking script 
        # can scale the embedding and lm_head independently.
        provider.share_embeddings_and_output_weights = False

        return provider

    def get_hf_tokenizer_kwargs(self) -> dict:
        """
        Ensure Megatron understands that Granite's <|end_of_text|> (ID 0)
        is the EOD token.
        """
        return {
            "eod_id": 0,
            "eos_id": 0,
            # Add bos_id if your model specifically uses one, 
            # though Granite config says add_bos_token is false.
        }

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map Megatron parameter names -> HF parameter names dynamically."""
        config = self.hf_config  # <-- The corrected rune
        
        # 1. Standard 1:1 Parameter mappings
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }

        # Dynamically add independent biases if the config calls for them
        if getattr(config, "attention_bias", False):
            param_mappings["decoder.layers.*.self_attention.linear_proj.bias"] = "model.layers.*.self_attn.o_proj.bias"
        
        if getattr(config, "mlp_bias", False):
            param_mappings["decoder.layers.*.mlp.linear_fc2.bias"] = "model.layers.*.mlp.down_proj.bias"

        mapping_list = [
            AutoMapping(megatron_param=meg, hf_param=hf) 
            for meg, hf in param_mappings.items()
        ]

        # 2. Handle concatenated weights (QKV and SwiGLU Gate/Up)
        mapping_list.extend([
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
        ])

        # 3. Handle concatenated biases if they exist
        if getattr(config, "attention_bias", False):
            mapping_list.append(
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.layers.*.self_attn.q_proj.bias",
                    k="model.layers.*.self_attn.k_proj.bias",
                    v="model.layers.*.self_attn.v_proj.bias",
                )
            )

        if getattr(config, "mlp_bias", False):
             mapping_list.append(
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.bias",
                    gate="model.layers.*.mlp.gate_proj.bias",
                    up="model.layers.*.mlp.up_proj.bias",
                )
            )

        return MegatronMappingRegistry(*mapping_list)