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
transmute_granite.py — Bake Granite HF scaling multipliers into checkpoint weights.

Granite models (ibm-granite family) apply three scaling factors at inference time
that are stored in the HuggingFace config.json rather than in the weight tensors:

  out = embed_tokens(ids) * embedding_multiplier
  residual += attn_out    * residual_multiplier   (o_proj output)
  residual += mlp_out     * residual_multiplier   (down_proj output)
  logits   = lm_head(h)  / logits_scaling

A standard Megatron-Core GPTModel has no knowledge of these config-level
scalars.  This script pre-processes the HuggingFace checkpoint by baking the
multipliers directly into the weights, then resetting the config values to 1.0.
The resulting checkpoint can be used directly with GraniteBridge (or any other
standard GPT-style loader) without further modification.

NOTE: GraniteBridge can also handle raw (non-transmuted) checkpoints by applying
the scaling on-the-fly during weight loading.  Running this script is therefore
OPTIONAL, but can be useful when you want the checkpoint itself to be self-
contained (e.g., for long-running training jobs that reload from a checkpoint).

Usage:
    python transmute_granite.py \\
        --source /path/to/granite-3.0-8b-base \\
        --target /path/to/granite-3.0-8b-base-transmuted
"""

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def transmute_granite(source_path: str, target_path: str) -> None:
    print(f"Loading config from {source_path} ...")
    config = AutoConfig.from_pretrained(source_path, trust_remote_code=True)

    m_e = float(getattr(config, "embedding_multiplier", 1.0))
    m_r = float(getattr(config, "residual_multiplier", 1.0))
    m_l = float(getattr(config, "logits_scaling", 1.0))
    m_a = float(getattr(config, "attention_multiplier", 1.0))
    tied = bool(getattr(config, "tie_word_embeddings", False))

    print(f"Multipliers  →  embedding: {m_e},  residual: {m_r},  logits_scaling: {m_l},  attention: {m_a}")
    print(f"tie_word_embeddings: {tied}")

    if m_e == 1.0 and m_r == 1.0 and m_l == 1.0 and m_a == 1.0:
        print("All multipliers are 1.0 — nothing to do.  Copying model as-is.")

    # Untie embeddings before loading so that lm_head gets its own tensor.
    # This is required so we can scale embed_tokens and lm_head independently.
    if tied:
        print("Untying embed_tokens / lm_head weights before transmutation ...")
        config.tie_word_embeddings = False

    print(f"Loading model weights from {source_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        source_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # If the weights were originally tied, lm_head.weight may still share
    # storage with embed_tokens.weight even after config.tie_word_embeddings=False.
    # Ensure lm_head has its own copy before modifying either tensor.
    if tied or (
        model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    ):
        model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight.clone())

    # ----- Bake embedding_multiplier into embed_tokens -----
    if m_e != 1.0:
        print(f"  embed_tokens.weight *= {m_e}")
        model.model.embed_tokens.weight.data.mul_(m_e)

    # ----- Bake residual_multiplier into o_proj and down_proj -----
    # NOTE: attention_multiplier is applied to attention *scores* (before softmax),
    # not to the attention output. Since softmax is non-linear, it CANNOT be baked
    # into o_proj weights. It must remain as a runtime config value.
    # Only residual_multiplier (applied after o_proj/down_proj) can be baked.
    if m_r != 1.0:
        print(f"  Scaling {len(model.model.layers)} layers' o_proj by {m_r} (res), down_proj by {m_r}")
        for layer in model.model.layers:
            layer.self_attn.o_proj.weight.data.mul_(m_r)
            if layer.self_attn.o_proj.bias is not None:
                layer.self_attn.o_proj.bias.data.mul_(m_r)

            layer.mlp.down_proj.weight.data.mul_(m_r)
            if layer.mlp.down_proj.bias is not None:
                layer.mlp.down_proj.bias.data.mul_(m_r)

    # ----- Bake logits_scaling into lm_head -----
    # The Granite forward pass computes:  logits = lm_head(hidden) / logits_scaling
    # We absorb this by dividing the lm_head weights.
    # Note: when originally tied AND embedding_multiplier != 1.0, we already
    # scaled embed_tokens above; lm_head is now independent, so only apply m_l.
    if m_l != 1.0:
        print(f"  lm_head.weight /= {m_l}")
        model.lm_head.weight.data.div_(m_l)

    # ----- Reset config multipliers to 1.0 (except attention_multiplier) -----
    # attention_multiplier cannot be baked — it scales attention scores before
    # softmax and must remain a runtime parameter.
    config.embedding_multiplier = 1.0
    config.residual_multiplier = 1.0
    config.logits_scaling = 1.0
    # config.attention_multiplier is intentionally NOT reset

    # ----- Save -----
    print(f"Saving transmuted model to {target_path} ...")
    model.save_pretrained(target_path)
    config.save_pretrained(target_path)
    print("Done. The transmuted checkpoint is ready for GraniteBridge / standard GPT loaders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bake Granite HuggingFace config multipliers into checkpoint weights.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the original HuggingFace Granite model directory.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path where the transmuted model will be saved.",
    )
    args = parser.parse_args()
    transmute_granite(args.source, args.target)
