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

import logging
import re
from typing import Dict, List, Mapping, Optional, Union

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import GraniteForCausalLM, GraniteMoeForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

from .granite_provider import GraniteMoeModelProvider, GraniteModelProvider

logger = logging.getLogger(__name__)


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
            # Must untie when multipliers differ (embedding_multiplier vs logits_scaling)
            # because Megatron can't apply different scaling to a shared weight.
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

        # attention_multiplier scales attention scores before softmax — it cannot
        # be baked into weights. Pass it as the softmax scaling factor to Megatron-Core.
        provider.softmax_scale = float(getattr(hf_config, "attention_multiplier", 1.0))

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
          attn_scores = (Q @ K^T) * attention_multiplier   (before softmax)
          hidden += o_proj(attn_out) * residual_multiplier
          hidden += mlp_out * residual_multiplier          (down_proj)
          logits  = lm_head(hidden) / logits_scaling

        We bake embedding_multiplier, residual_multiplier, and logits_scaling
        into the corresponding weight tensors. attention_multiplier CANNOT be
        baked because it scales attention scores before softmax (non-linear),
        and must be handled separately by the Megatron attention layer.

        For pre-transmuted weights the multipliers are already 1.0 (or very
        close), so the multiplication/division is effectively a no-op.
        """
        # Handle tied weights: lm_head.weight may not exist in HF state dict
        # when tie_word_embeddings=True. Fall back to embed_tokens.weight.
        if isinstance(hf_param, str) and hf_param == "lm_head.weight" and hf_param not in hf_state_dict:
            hf_param = "model.embed_tokens.weight"
            weight = super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)
            # Apply logits_scaling (not embedding_multiplier) to this copy
            m_e, m_r, m_l, m_a = self._get_granite_multipliers()
            if m_l != 1.0:
                original_dtype = weight.dtype
                w = weight.detach().float() / m_l
                return w.to(original_dtype)
            return weight

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
            w = w / m_l

        elif re.search(r"self_attn\.o_proj\.weight$", hf_param):
            # Only residual_multiplier can be baked into o_proj.
            # attention_multiplier scales attention *scores* (before softmax),
            # not the output, so it cannot be absorbed into any weight.
            w = w * m_r

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
        no_scaling = (m_e == 1.0 and m_r == 1.0 and m_l == 1.0 and m_a == 1.0)

        result = {}
        for hf_key, weight in converted_weights_dict.items():
            # Use the original HF weight's dtype when available, so the exported
            # checkpoint matches what HF originally had (e.g. bfloat16).
            target_dtype = hf_state_dict[hf_key].dtype if hf_key in hf_state_dict else weight.dtype

            if no_scaling:
                result[hf_key] = weight.to(target_dtype)
                continue

            w = weight.detach().float()

            if hf_key == "model.embed_tokens.weight":
                w = w / m_e
            elif hf_key == "lm_head.weight":
                w = w * m_l
            elif re.search(r"self_attn\.o_proj\.weight$", hf_key):
                # Inverse of residual_multiplier only (attention_multiplier is not baked)
                w = w / m_r
            elif re.search(r"self_attn\.o_proj\.bias$", hf_key):
                # Bias only had residual_multiplier applied
                w = w / m_r
            elif re.search(r"mlp\.down_proj\.(weight|bias)$", hf_key):
                w = w / m_r
            else:
                result[hf_key] = weight.to(target_dtype)
                continue

            result[hf_key] = w.to(target_dtype)

        return result

    # ------------------------------------------------------------------
    # Weight mapping
    # ------------------------------------------------------------------

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map Megatron parameter names → HF parameter names."""
        config = self.hf_config
        attention_bias = getattr(config, "attention_bias", False)
        mlp_bias = getattr(config, "mlp_bias", False)
        tied = getattr(config, "tie_word_embeddings", False)

        # ---- 1:1 parameter mappings ----
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }

        # When HF has tied weights, lm_head.weight doesn't exist in the checkpoint.
        # We map to lm_head.weight with allow_hf_name_mismatch=True so
        # build_conversion_tasks doesn't skip it. maybe_modify_loaded_hf_weight
        # handles the fallback to model.embed_tokens.weight and applies / logits_scaling.
        if tied:
            lm_head_mapping = AutoMapping(
                megatron_param="output_layer.weight",
                hf_param="lm_head.weight",
            )
            lm_head_mapping.allow_hf_name_mismatch = True
        else:
            lm_head_mapping = None
            param_mappings["output_layer.weight"] = "lm_head.weight"

        param_mappings.update({
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
        })

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

        # Add the tied lm_head mapping if needed
        if lm_head_mapping is not None:
            mapping_list.append(lm_head_mapping)

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


@MegatronModelBridge.register_bridge(
    source=GraniteMoeForCausalLM,
    target=GPTModel,
    provider=GraniteMoeModelProvider,
    model_type="granitemoe",
)
class GraniteMoeBridge(GraniteBridge):
    """
    Megatron Bridge for Granite MoE Causal LM.

    Extends GraniteBridge with Mixture-of-Experts support. Inherits all Granite
    scaling multiplier handling (embedding_multiplier, residual_multiplier,
    logits_scaling, attention_multiplier).

    The HF GraniteMoe architecture uses:
      - block_sparse_moe.router.layer.weight: top-k gating linear
      - block_sparse_moe.input_linear.weight: [num_experts, ffn*2, hidden] (fused gate+up)
      - block_sparse_moe.output_linear.weight: [num_experts, hidden, ffn] (down proj)

    Supported models:
      - ibm/PowerMoE-3b
      - Any GraniteMoeForCausalLM variant
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GraniteMoeModelProvider:
        """Convert HuggingFace GraniteMoe config to a GraniteMoeModelProvider."""
        hf_config = hf_pretrained.config

        provider = GraniteMoeModelProvider(
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
            share_embeddings_and_output_weights=False,
            add_qkv_bias=getattr(hf_config, "attention_bias", False),
            add_bias_linear=getattr(hf_config, "mlp_bias", False),
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            kv_channels=getattr(hf_config, "head_dim", None),
            # MoE fields
            num_moe_experts=hf_config.num_local_experts,
            moe_router_topk=hf_config.num_experts_per_tok,
            moe_ffn_hidden_size=hf_config.intermediate_size,
            moe_aux_loss_coeff=getattr(hf_config, "router_aux_loss_coef", 0.001),
        )

        # attention_multiplier → softmax_scale (same as dense Granite)
        provider.softmax_scale = float(getattr(hf_config, "attention_multiplier", 1.0))

        return provider

    # ------------------------------------------------------------------
    # Multiplier handling (extends dense Granite for MoE output_linear)
    # ------------------------------------------------------------------

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict, hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | dict:
        """
        Bake Granite scaling multipliers during HF → Megatron loading.

        Extends the dense Granite logic to handle the MoE output_linear weight,
        which receives the residual_multiplier (same role as dense down_proj).
        """
        if isinstance(hf_param, str) and re.search(r"block_sparse_moe\.output_linear\.weight$", hf_param):
            weight = super(GraniteBridge, self).maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)
            m_e, m_r, m_l, m_a = self._get_granite_multipliers()
            if m_r != 1.0:
                original_dtype = weight.dtype
                w = weight.detach().float() * m_r
                return w.to(original_dtype)
            return weight

        # All other weights handled by dense GraniteBridge
        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Un-bake Granite multipliers during Megatron → HF export.

        Extends the dense Granite logic to handle the MoE output_linear weight.
        """
        m_e, m_r, m_l, m_a = self._get_granite_multipliers()

        # First let the dense bridge handle standard weights
        result = super().maybe_modify_converted_hf_weight(task, converted_weights_dict, hf_state_dict)

        # Then fix up MoE output_linear (residual_multiplier inverse)
        if m_r != 1.0:
            for hf_key, weight in list(result.items()):
                if re.search(r"block_sparse_moe\.output_linear\.weight$", hf_key):
                    target_dtype = hf_state_dict[hf_key].dtype if hf_key in hf_state_dict else weight.dtype
                    w = weight.detach().float() / m_r
                    result[hf_key] = w.to(target_dtype)

        return result

    # ------------------------------------------------------------------
    # Expert weight loading (manual, outside the mapping system)
    #
    # HF GraniteMoe stores expert weights as stacked tensors:
    #   input_linear.weight: [num_experts, ffn*2, hidden]
    #   output_linear.weight: [num_experts, hidden, ffn]
    #
    # Megatron TEGroupedMLP stores per-expert parameters:
    #   experts.linear_fc1.weight0, weight1, ..., weight{N-1}
    #   experts.linear_fc2.weight0, weight1, ..., weight{N-1}
    #
    # The standard mapping system can't handle this stacked↔per-expert
    # conversion, so we do it manually after the standard loading.
    # ------------------------------------------------------------------

    def load_weights_hf_to_megatron(
        self,
        hf_pretrained,
        megatron_model,
        allowed_mismatched_params: Optional[List[str]] = None,
    ):
        """Load HF weights into Megatron, with manual expert weight handling.

        Expert weights are handled separately because HF GraniteMoe stores them
        as stacked tensors [num_experts, ...] while Megatron TEGroupedMLP stores
        them as per-expert parameters (weight0, weight1, ...).
        """
        import contextlib

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        # Phase 1: Standard mapping for non-expert weights
        with contextlib.ExitStack() as stack:
            if hasattr(megatron_model[0], "hide_teacher_model"):
                stack.enter_context(megatron_model[0].hide_teacher_model())
            if hasattr(megatron_model[0], "hide_loss_modules"):
                stack.enter_context(megatron_model[0].hide_loss_modules())
            hf_to_megatron_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)

        hf_state_dict = hf_pretrained.state if hasattr(hf_pretrained, "state") else {}

        description = f"Loading from {hf_pretrained.model_name_or_path}"
        skipped_none = []
        loaded_params = []
        for task in self._with_progress_tracking(hf_to_megatron_tasks, description):
            # Skip None tasks (unmatched expert params)
            if task is None:
                skipped_none.append("None task")
                continue
            if task.megatron_module is None:
                skipped_none.append(f"no module: {task.mapping.megatron_param}")
                continue
            loaded_params.append(task.mapping.megatron_param)
            hf_weights = self.maybe_modify_loaded_hf_weight(task.mapping.hf_param, hf_state_dict)
            converted_weights = task.mapping.hf_to_megatron(hf_weights, task.megatron_module)
            if converted_weights is not None:
                assert task.param_weight is not None
                if converted_weights.shape != task.param_weight.shape:
                    raise ValueError(
                        f"Shape mismatch: {task.mapping.megatron_param} "
                        f"expected {task.param_weight.shape}, got {converted_weights.shape}"
                    )
                task.param_weight.data.copy_(converted_weights)

        logger.info(f"GraniteMoE: standard mapping loaded {len(loaded_params)} params, skipped {len(skipped_none)}")
        if skipped_none:
            logger.info(f"GraniteMoE: skipped tasks: {skipped_none[:20]}")
        logger.info(f"GraniteMoE: loaded params: {loaded_params[:20]}")

        # Phase 2: Manual expert weight loading
        # With EP, each rank holds a subset of experts. Local weight indices
        # (weight0, weight1, ...) must be mapped to global expert indices:
        #   global_idx = ep_rank * num_local_experts + local_idx
        from megatron.core import parallel_state

        m_e, m_r, m_l, m_a = self._get_granite_multipliers()
        num_local_experts = self.hf_config.num_local_experts
        try:
            ep_rank = parallel_state.get_expert_model_parallel_rank()
            ep_size = parallel_state.get_expert_model_parallel_world_size()
            num_local_experts_per_rank = num_local_experts // ep_size
        except Exception:
            ep_rank = 0
            num_local_experts_per_rank = num_local_experts

        # TP splitting: expert weights are sharded across TP ranks
        try:
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
        except Exception:
            tp_rank = 0
            tp_size = 1

        expert_loaded_count = 0
        expert_names_seen = []
        for model in megatron_model:
            for name, param in model.named_parameters():
                if "expert" in name and "weight" in name and "layers.0." in name:
                    expert_names_seen.append(name)

                # Match expert FC1 weights: decoder.layers.N.mlp.experts.linear_fc1.weightI
                fc1_match = re.match(
                    r".*?decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc1\.weight(\d+)$", name
                )
                if fc1_match:
                    layer_idx = int(fc1_match.group(1))
                    local_expert_idx = int(fc1_match.group(2))
                    global_expert_idx = ep_rank * num_local_experts_per_rank + local_expert_idx
                    hf_key = f"model.layers.{layer_idx}.block_sparse_moe.input_linear.weight"
                    if hf_key in hf_state_dict:
                        expert_weight = hf_state_dict[hf_key][global_expert_idx]
                        # FC1 (input_linear): output dim (dim 0) split by TP
                        if tp_size > 1:
                            chunk_size = expert_weight.shape[0] // tp_size
                            expert_weight = expert_weight[tp_rank * chunk_size : (tp_rank + 1) * chunk_size]
                        param.data.copy_(expert_weight)
                        expert_loaded_count += 1
                    continue

                # Match expert FC2 weights: decoder.layers.N.mlp.experts.linear_fc2.weightI
                fc2_match = re.match(
                    r".*?decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc2\.weight(\d+)$", name
                )
                if fc2_match:
                    layer_idx = int(fc2_match.group(1))
                    local_expert_idx = int(fc2_match.group(2))
                    global_expert_idx = ep_rank * num_local_experts_per_rank + local_expert_idx
                    hf_key = f"model.layers.{layer_idx}.block_sparse_moe.output_linear.weight"
                    if hf_key in hf_state_dict:
                        expert_weight = hf_state_dict[hf_key][global_expert_idx]
                        # FC2 (output_linear): input dim (dim 1) split by TP
                        if tp_size > 1:
                            chunk_size = expert_weight.shape[1] // tp_size
                            expert_weight = expert_weight[:, tp_rank * chunk_size : (tp_rank + 1) * chunk_size]
                        # Apply residual_multiplier (same as dense down_proj)
                        if m_r != 1.0:
                            original_dtype = expert_weight.dtype
                            expert_weight = (expert_weight.detach().float() * m_r).to(original_dtype)
                        param.data.copy_(expert_weight)
                        expert_loaded_count += 1
                    continue

        logger.info(f"GraniteMoE: expert param names (layer 0): {expert_names_seen[:10]}")
        logger.info(f"GraniteMoE: total expert weights loaded in Phase 2: {expert_loaded_count}")
        logger.info(
            f"GraniteMoE: expert weights loaded (ep_rank={ep_rank}, "
            f"{num_local_experts_per_rank} local experts, "
            f"global offset={ep_rank * num_local_experts_per_rank})"
        )
        return megatron_model

    # ------------------------------------------------------------------
    # Weight mapping (non-expert params only)
    # ------------------------------------------------------------------

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map Megatron parameter names → HF parameter names for GraniteMoe.

        Expert weights (linear_fc1, linear_fc2) are handled separately in
        load_weights_hf_to_megatron because HF stores them as stacked tensors
        while Megatron stores them as per-expert parameters.
        """
        config = self.hf_config
        attention_bias = getattr(config, "attention_bias", False)
        tied = getattr(config, "tie_word_embeddings", False)

        # ---- 1:1 parameter mappings (no expert weights) ----
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Pre-attention layernorm (fused with QKV in TE)
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": (
                "model.layers.*.input_layernorm.weight"
            ),
            # Pre-MLP layernorm (separate for MoE, not fused with FC1)
            "decoder.layers.*.pre_mlp_layernorm.weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),
            # Attention output projection
            "decoder.layers.*.self_attention.linear_proj.weight": (
                "model.layers.*.self_attn.o_proj.weight"
            ),
            # MoE router
            "decoder.layers.*.mlp.router.weight": (
                "model.layers.*.block_sparse_moe.router.layer.weight"
            ),
        }

        # Tied embeddings handling (same as dense Granite)
        if tied:
            lm_head_mapping = AutoMapping(
                megatron_param="output_layer.weight",
                hf_param="lm_head.weight",
            )
            lm_head_mapping.allow_hf_name_mismatch = True
        else:
            lm_head_mapping = None
            param_mappings["output_layer.weight"] = "lm_head.weight"

        # Optional attention bias
        if attention_bias:
            param_mappings["decoder.layers.*.self_attention.linear_proj.bias"] = (
                "model.layers.*.self_attn.o_proj.bias"
            )

        mapping_list = [
            AutoMapping(megatron_param=meg, hf_param=hf)
            for meg, hf in param_mappings.items()
        ]

        if lm_head_mapping is not None:
            mapping_list.append(lm_head_mapping)

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

        return MegatronMappingRegistry(*mapping_list)