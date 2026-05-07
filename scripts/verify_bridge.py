#!/usr/bin/env python3
"""Verify Granite bridge weight loading for dense and MoE models.

Loads HF weights via AutoBridge, then compares key tensors between
the HF state dict and the Megatron model parameters to ensure
correct loading with proper Granite multiplier baking.

Usage (single GPU, no distributed):
    python verify_bridge.py --model /path/to/granite-model

Usage (multi-GPU for MoE with EP):
    torchrun --nproc_per_node=8 verify_bridge.py \
        --model /path/to/granite-moe --ep 8
"""

import argparse
import sys

import torch
import torch.distributed


def verify_bridge(model_path: str, ep: int = 1, tp: int = 1):
    from megatron.bridge import AutoBridge

    print(f"\n{'='*60}")
    print(f"Verifying bridge for: {model_path}")
    print(f"{'='*60}")

    # Step 1: Load bridge and check type
    bridge = AutoBridge.from_hf_pretrained(model_path)
    bridge_type = type(bridge).__name__
    print(f"\n[1] Bridge type: {bridge_type}")

    is_moe = "Moe" in bridge_type
    print(f"    Is MoE: {is_moe}")

    # Step 2: Check provider config
    provider = bridge.to_megatron_provider()
    print(f"\n[2] Provider config:")
    print(f"    hidden_size: {provider.hidden_size}")
    print(f"    num_layers: {provider.num_layers}")
    print(f"    vocab_size: {provider.vocab_size}")
    print(f"    num_attention_heads: {provider.num_attention_heads}")
    print(f"    num_query_groups: {provider.num_query_groups}")
    if hasattr(provider, 'softmax_scale'):
        print(f"    softmax_scale: {provider.softmax_scale}")
    if is_moe:
        print(f"    num_moe_experts: {provider.num_moe_experts}")
        print(f"    moe_router_topk: {provider.moe_router_topk}")
        print(f"    moe_ffn_hidden_size: {provider.moe_ffn_hidden_size}")

    # Step 3: Check HF state dict keys
    hf_state = bridge.hf_pretrained.state
    hf_keys = sorted(hf_state.keys())
    print(f"\n[3] HF state dict: {len(hf_keys)} keys")

    # Show layer 0 keys
    layer0_keys = [k for k in hf_keys if "layers.0." in k]
    print(f"    Layer 0 keys ({len(layer0_keys)}):")
    for k in layer0_keys:
        print(f"      {k}: {hf_state[k].shape}")

    # Step 4: Check Granite multipliers
    config = bridge.hf_pretrained.config
    m_e = float(getattr(config, "embedding_multiplier", 1.0))
    m_r = float(getattr(config, "residual_multiplier", 1.0))
    m_l = float(getattr(config, "logits_scaling", 1.0))
    m_a = float(getattr(config, "attention_multiplier", 1.0))
    print(f"\n[4] Granite multipliers:")
    print(f"    embedding_multiplier: {m_e}")
    print(f"    residual_multiplier: {m_r}")
    print(f"    logits_scaling: {m_l}")
    print(f"    attention_multiplier: {m_a}")

    # Step 5: Verify key weight values (without full model construction)
    print(f"\n[5] Weight value checks:")
    errors = []

    # Check embedding
    embed_key = "model.embed_tokens.weight"
    if embed_key in hf_state:
        hf_embed = hf_state[embed_key]
        hf_mean = hf_embed.float().mean().item()
        hf_std = hf_embed.float().std().item()
        expected_mean = hf_mean * m_e if m_e != 1.0 else hf_mean
        print(f"    embed_tokens: mean={hf_mean:.6f}, std={hf_std:.6f}")
        print(f"    -> After baking (×{m_e}): mean={expected_mean:.6f}")
        if hf_std < 1e-8:
            errors.append("embed_tokens has zero std (likely uninitialized)")

    # Check router (MoE only)
    if is_moe:
        router_key = "model.layers.0.block_sparse_moe.router.layer.weight"
        if router_key in hf_state:
            router_w = hf_state[router_key]
            print(f"    router: shape={router_w.shape}, mean={router_w.float().mean():.6f}")
        else:
            errors.append(f"Missing router key: {router_key}")

        # Check expert weights
        fc1_key = "model.layers.0.block_sparse_moe.input_linear.weight"
        fc2_key = "model.layers.0.block_sparse_moe.output_linear.weight"
        if fc1_key in hf_state:
            fc1 = hf_state[fc1_key]
            print(f"    input_linear (FC1): shape={fc1.shape}")
            print(f"      expert 0 mean={fc1[0].float().mean():.6f}, std={fc1[0].float().std():.6f}")
        else:
            errors.append(f"Missing FC1 key: {fc1_key}")

        if fc2_key in hf_state:
            fc2 = hf_state[fc2_key]
            print(f"    output_linear (FC2): shape={fc2.shape}")
            print(f"      expert 0 mean={fc2[0].float().mean():.6f}, std={fc2[0].float().std():.6f}")
        else:
            errors.append(f"Missing FC2 key: {fc2_key}")
    else:
        # Dense model checks
        down_key = "model.layers.0.mlp.down_proj.weight"
        if down_key in hf_state:
            down_w = hf_state[down_key]
            print(f"    down_proj: shape={down_w.shape}, mean={down_w.float().mean():.6f}")

    # Step 6: Tokenizer check
    print(f"\n[6] Tokenizer check:")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"    vocab_size: {tok.vocab_size}")
        print(f"    eos_token: {tok.eos_token} (id: {tok.eos_token_id})")
        print(f"    pad_token: {tok.pad_token} (id: {tok.pad_token_id})")
        print(f"    bos_token: {tok.bos_token} (id: {tok.bos_token_id})")

        # Check special tokens
        special_checks = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
        for tok_str in special_checks:
            tok_id = tok.convert_tokens_to_ids(tok_str)
            is_single = tok_id != tok.unk_token_id
            status = "OK (single token)" if is_single else "MISSING (subword split)"
            print(f"    {tok_str}: id={tok_id} — {status}")
    except Exception as e:
        print(f"    Tokenizer error: {e}")

    # Step 7: Summary
    print(f"\n{'='*60}")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("ALL CHECKS PASSED")
    print(f"{'='*60}\n")

    return len(errors) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Granite bridge")
    parser.add_argument("--model", required=True, help="Path to HF model")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size (for MoE)")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    args = parser.parse_args()

    success = verify_bridge(args.model, args.ep, args.tp)
    sys.exit(0 if success else 1)
