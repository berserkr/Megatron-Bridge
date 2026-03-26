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
Megatron Bridge for IBM Granite Causal Language Models.

Granite models store scaling multipliers in the HuggingFace config
(embedding_multiplier, residual_multiplier, logits_scaling) rather than
pre-multiplied into the weights. This bridge handles baking those multipliers
into the weights transparently during HF → Megatron loading, matching what
transmute_granite.py does as a standalone pre-processing step.

If you run transmute_granite.py first (which bakes the multipliers and resets
the config values to 1.0), this bridge will detect the no-op case and skip
the scaling step at zero cost.

Supported models (tested):
  - ibm-granite/granite-3.0-3b-a800m-base / instruct
  - ibm-granite/granite-3.0-8b-base / instruct
  - ibm-granite/granite-3.1-8b-base / instruct
  - Any Granite variant with GraniteForCausalLM architecture

Reference:
  https://huggingface.co/ibm-granite
"""

import re
from typing import Dict, Mapping, Optional

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import GraniteForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
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

    Handles Granite-specific weight mapping and the transparent baking of
    HuggingFace scaling multipliers into Megatron weights.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GraniteModelProvider:
        """Convert HuggingFace Granite config to a GraniteModelProvider."""
        hf_config = hf_pretrained.config

        provider = GraniteModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
            seq_length=hf_config.max_position_embeddings,
            vocab_size=hf_config.vocab_size,
            layernorm_epsilon=hf_config.rms_norm_eps,
            rotary_base=rope_theta_from_hf(hf_config),
            init_method_std=hf_config.initializer_range,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            # Granite always untied: lm_head and embed_tokens are scaled independently
            share_embeddings_and_output_weights=False,
            # Attention biases (False for most Granite models)
            add_qkv_bias=getattr(hf_config, "attention_bias", False),
            # MLP and output-projection biases are separate; add_bias_linear covers
            # both MLP *and* output projections in Megatron, so only enable it if
            # the model has mlp_bias set (rare).
            add_bias_linear=getattr(hf_config, "mlp_bias", False),
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            kv_channels=getattr(hf_config, "head_dim", None),
        )

        return provider

    # ------------------------------------------------------------------
    # Granite scaling multiplier handling
    # ------------------------------------------------------------------

    def _get_granite_multipliers(self):
        """
        Read Granite scaling multipliers from the loaded HF config.

        Raw Granite checkpoints store these in config.json.  After running
        transmute_granite.py the values are reset to 1.0, so this method
        returns (1.0, 1.0, 1.0, 1.0) for pre-transmuted weights, causing a no-op.

        Returns:
            (embedding_multiplier, residual_multiplier, logits_scaling, attention_multiplier)
        """
        config = self.hf_config
        m_e = float(getattr(config, "embedding_multiplier", 1.0))
        m_r = float(getattr(config, "residual_multiplier", 1.0))
        m_l = float(getattr(config, "logits_scaling", 1.0))
        m_a = float(getattr(config, "attention_multiplier", 1.0))
        return m_e, m_r, m_l, m_a

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict, hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | dict:
        """
        Bake Granite's scaling multipliers into the weights on-the-fly during
        HF → Megatron loading.

        Granite's forward pass applies scaling factors stored in the config
        rather than the weights:

          output  = embed_tokens(token_ids) * embedding_multiplier
          attn_out = attn(hidden) * attention_multiplier   (before o_proj)
          hidden += o_proj(attn_out) * residual_multiplier
          hidden += mlp_out * residual_multiplier          (down_proj)
          logits  = lm_head(hidden) / logits_scaling

        We bake these into the corresponding weight tensors so that
        Megatron's standard GPTModel (which has no knowledge of these
        config-level scalars) produces identical results.

        For o_proj, the combined scaling is attention_multiplier * residual_multiplier
        because attention_multiplier is applied before the linear projection
        and residual_multiplier after it.  Since o_proj is linear:
          o_proj(m_a * x) * m_r = (m_a * m_r) * o_proj(x)   [when no bias]

        For pre-transmuted weights the multipliers are already 1.0 (or very
        close), so the multiplication/division is effectively a no-op.
        """
        weight = super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

        m_e, m_r, m_l, m_a = self._get_granite_multipliers()

        # Fast path: all multipliers are trivial
        if m_e == 1.0 and m_r == 1.0 and m_l == 1.0 and m_a == 1.0:
            return weight

        if not isinstance(hf_param, str):
            # QKV / GatedMLP dict mappings — these are not scaled by Granite's
            # multipliers, so pass through unchanged.
            return weight

        original_dtype = weight.dtype
        w = weight.detach().float()

        if hf_param == "model.embed_tokens.weight":
            # embed_tokens output is scaled by embedding_multiplier
            w = w * m_e

        elif hf_param == "lm_head.weight":
            # logits are divided by logits_scaling — so the weight gets divided
            # (we keep the multiplier convention: weight /= logits_scaling)
            w = w / m_l

        elif re.search(r"self_attn\.o_proj\.weight$", hf_param):
            # attention_multiplier scales attn output before o_proj (linear),
            # residual_multiplier scales o_proj output before residual add.
            # Combined: o_proj.weight *= attention_multiplier * residual_multiplier
            w = w * m_a * m_r

        elif re.search(r"self_attn\.o_proj\.bias$", hf_param):
            # Bias is only scaled by residual_multiplier (applied after o_proj)
            w = w * m_r

        elif re.search(r"mlp\.down_proj\.weight$", hf_param):
            # MLP residual contribution is scaled by residual_multiplier
            w = w * m_r

        elif re.search(r"mlp\.down_proj\.bias$", hf_param):
            w = w * m_r

        else:
            return weight

        return w.to(original_dtype)

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Un-bake the Granite multipliers when converting Megatron → HF.

        This is the inverse of maybe_modify_loaded_hf_weight: weights that were
        scaled during loading are unscaled here so the exported HF checkpoint is
        consistent with the original config.
        """
        m_e, m_r, m_l, m_a = self._get_granite_multipliers()

        if m_e == 1.0 and m_r == 1.0 and m_l == 1.0 and m_a == 1.0:
            return converted_weights_dict

        result = {}
        for hf_key, weight in converted_weights_dict.items():
            original_dtype = weight.dtype
            w = weight.detach().float()

            if hf_key == "model.embed_tokens.weight":
                w = w / m_e
            elif hf_key == "lm_head.weight":
                w = w * m_l
            elif re.search(r"self_attn\.o_proj\.weight$", hf_key):
                # Inverse of combined attention_multiplier * residual_multiplier
                w = w / (m_a * m_r)
            elif re.search(r"self_attn\.o_proj\.bias$", hf_key):
                # Bias only had residual_multiplier applied
                w = w / m_r
            elif re.search(r"mlp\.down_proj\.(weight|bias)$", hf_key):
                w = w / m_r
            else:
                result[hf_key] = weight
                continue

            result[hf_key] = w.to(original_dtype)

        return result

    # ------------------------------------------------------------------
    # Weight mapping
    # ------------------------------------------------------------------

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map Megatron parameter names → HF parameter names."""
        config = self.hf_config
        attention_bias = getattr(config, "attention_bias", False)
        mlp_bias = getattr(config, "mlp_bias", False)

        # ---- 1:1 parameter mappings ----
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Per-layer pre-attention and pre-MLP layer norms (fused with linear in TE)
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": (
                "model.layers.*.input_layernorm.weight"
            ),
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),
            # Attention output projection
            "decoder.layers.*.self_attention.linear_proj.weight": (
                "model.layers.*.self_attn.o_proj.weight"
            ),
            # MLP down projection
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }

        # Optional per-layer output projection bias (attention_bias in HF config)
        if attention_bias:
            param_mappings["decoder.layers.*.self_attention.linear_proj.bias"] = (
                "model.layers.*.self_attn.o_proj.bias"
            )

        # Optional MLP down-projection bias (mlp_bias in HF config)
        if mlp_bias:
            param_mappings["decoder.layers.*.mlp.linear_fc2.bias"] = (
                "model.layers.*.mlp.down_proj.bias"
            )

        mapping_list = [
            AutoMapping(megatron_param=meg, hf_param=hf)
            for meg, hf in param_mappings.items()
        ]

        # ---- QKV concatenation ----
        mapping_list.append(
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            )
        )

        if attention_bias:
            mapping_list.append(
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.layers.*.self_attn.q_proj.bias",
                    k="model.layers.*.self_attn.k_proj.bias",
                    v="model.layers.*.self_attn.v_proj.bias",
                )
            )

        # ---- SwiGLU gate+up projection ----
        mapping_list.append(
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            )
        )

        if mlp_bias:
            mapping_list.append(
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.bias",
                    gate="model.layers.*.mlp.gate_proj.bias",
                    up="model.layers.*.mlp.up_proj.bias",
                )
            )

        return MegatronMappingRegistry(*mapping_list)