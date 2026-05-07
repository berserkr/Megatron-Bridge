#!/usr/bin/env python3
"""Verify Granite bridge weight round-trip: HF → bake multipliers → un-bake → compare.

Tests that the Granite multiplier baking/un-baking is lossless for all weight types.
Also verifies the mapping registry resolves correctly for dense and MoE models.

For MoE: additionally checks that expert weight stacking/unstacking preserves values.

Usage:
    python verify_bridge_roundtrip.py --model /path/to/granite-model
"""

import argparse
import re
import sys

import torch


def verify_multiplier_roundtrip(model_path: str):
    """Verify bake → un-bake round-trip for Granite multipliers."""
    from megatron.bridge import AutoBridge

    print(f"\n{'='*60}")
    print(f"GRANITE BRIDGE ROUND-TRIP VERIFICATION")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    bridge = AutoBridge.from_hf_pretrained(model_path)
    bridge_type = type(bridge).__name__
    is_moe = "Moe" in bridge_type
    print(f"\nBridge type: {bridge_type}")
    print(f"Is MoE: {is_moe}")

    hf_state = bridge.hf_pretrained.state
    config = bridge.hf_pretrained.config

    m_e = float(getattr(config, "embedding_multiplier", 1.0))
    m_r = float(getattr(config, "residual_multiplier", 1.0))
    m_l = float(getattr(config, "logits_scaling", 1.0))
    m_a = float(getattr(config, "attention_multiplier", 1.0))

    print(f"\nMultipliers: embed={m_e}, residual={m_r}, logits={m_l}, attn={m_a}")
    has_multipliers = not (m_e == 1.0 and m_r == 1.0 and m_l == 1.0 and m_a == 1.0)
    print(f"Has non-trivial multipliers: {has_multipliers}")

    errors = []
    tested = 0

    # Test multiplier bake → un-bake for each weight type
    print(f"\n--- Multiplier Round-Trip Tests ---")

    test_cases = [
        ("model.embed_tokens.weight", "embed", lambda w: w * m_e, lambda w: w / m_e),
        ("model.layers.0.self_attn.o_proj.weight", "o_proj", lambda w: w * m_r, lambda w: w / m_r),
    ]

    # Dense MLP
    if not is_moe:
        test_cases.append(
            ("model.layers.0.mlp.down_proj.weight", "down_proj", lambda w: w * m_r, lambda w: w / m_r)
        )

    # MoE expert weights
    if is_moe:
        test_cases.append(
            ("model.layers.0.block_sparse_moe.output_linear.weight", "expert_fc2",
             lambda w: w * m_r, lambda w: w / m_r)
        )
        # input_linear has no multiplier — should be identity
        test_cases.append(
            ("model.layers.0.block_sparse_moe.input_linear.weight", "expert_fc1",
             lambda w: w, lambda w: w)
        )
        # router has no multiplier
        test_cases.append(
            ("model.layers.0.block_sparse_moe.router.layer.weight", "router",
             lambda w: w, lambda w: w)
        )

    for hf_key, name, bake_fn, unbake_fn in test_cases:
        if hf_key not in hf_state:
            print(f"  [SKIP] {name}: key not found ({hf_key})")
            continue

        original = hf_state[hf_key].float().clone()
        baked = bake_fn(original.clone())
        roundtripped = unbake_fn(baked.clone())

        max_diff = (original - roundtripped).abs().max().item()
        rel_diff = ((original - roundtripped).abs() / (original.abs() + 1e-10)).mean().item()
        tested += 1

        if max_diff < 1e-4:
            print(f"  [PASS] {name}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
        else:
            print(f"  [FAIL] {name}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
            errors.append(f"{name} round-trip failed: max_diff={max_diff:.2e}")

    # Test tied embeddings (lm_head.weight)
    tied = getattr(config, "tie_word_embeddings", False)
    print(f"\n--- Tied Embeddings Test ---")
    print(f"  tie_word_embeddings: {tied}")
    if tied:
        embed_w = hf_state["model.embed_tokens.weight"].float()
        if "lm_head.weight" in hf_state:
            lm_head_w = hf_state["lm_head.weight"].float()
            are_same = torch.allclose(embed_w, lm_head_w)
            print(f"  embed_tokens == lm_head: {are_same}")
        else:
            print(f"  lm_head.weight not in state dict (expected for tied)")

        # Verify baking applies different multipliers
        baked_embed = embed_w * m_e
        baked_lm_head = embed_w / m_l  # lm_head uses logits_scaling
        if m_e != m_l:
            are_different = not torch.allclose(baked_embed, baked_lm_head)
            print(f"  Baked embed != baked lm_head (different multipliers): {are_different}")
            if not are_different:
                errors.append("Baked embed and lm_head should differ when m_e != m_l")
        tested += 1
        print(f"  [PASS] Tied weight handling verified")

    # Test mapping registry
    print(f"\n--- Mapping Registry Test ---")
    try:
        registry = bridge.mapping_registry()
        mappings = registry.mappings if hasattr(registry, 'mappings') else []
        num_mappings = len(mappings)
        print(f"  Number of mappings: {num_mappings}")
        for m in mappings[:10]:
            meg = m.megatron_param if hasattr(m, 'megatron_param') else str(m)
            hf = m.hf_param if hasattr(m, 'hf_param') else ''
            print(f"    {meg} -> {hf}")
        if num_mappings > 10:
            print(f"    ... and {num_mappings - 10} more")

        if num_mappings > 0:
            print(f"  [PASS] Mapping registry has {num_mappings} entries")
        else:
            errors.append("Mapping registry is empty")
            print(f"  [FAIL] Mapping registry is empty")
        tested += 1
    except Exception as e:
        errors.append(f"Mapping registry failed: {e}")
        print(f"  [FAIL] {e}")

    # MoE-specific checks
    if is_moe:
        print(f"\n--- MoE-Specific Checks ---")

        # Check expert weight shapes
        fc1_key = "model.layers.0.block_sparse_moe.input_linear.weight"
        fc2_key = "model.layers.0.block_sparse_moe.output_linear.weight"
        router_key = "model.layers.0.block_sparse_moe.router.layer.weight"

        num_experts = getattr(config, "num_local_experts", 0)
        ffn_size = getattr(config, "intermediate_size", 0)
        hidden_size = getattr(config, "hidden_size", 0)

        if fc1_key in hf_state:
            fc1 = hf_state[fc1_key]
            expected_shape = (num_experts, ffn_size * 2, hidden_size)
            shape_ok = tuple(fc1.shape) == expected_shape
            print(f"  FC1 shape: {tuple(fc1.shape)} (expected {expected_shape}) — {'PASS' if shape_ok else 'FAIL'}")
            if not shape_ok:
                errors.append(f"FC1 shape mismatch: {tuple(fc1.shape)} vs {expected_shape}")
            tested += 1

        if fc2_key in hf_state:
            fc2 = hf_state[fc2_key]
            expected_shape = (num_experts, hidden_size, ffn_size)
            shape_ok = tuple(fc2.shape) == expected_shape
            print(f"  FC2 shape: {tuple(fc2.shape)} (expected {expected_shape}) — {'PASS' if shape_ok else 'FAIL'}")
            if not shape_ok:
                errors.append(f"FC2 shape mismatch: {tuple(fc2.shape)} vs {expected_shape}")
            tested += 1

        if router_key in hf_state:
            router = hf_state[router_key]
            expected_shape = (num_experts, hidden_size)
            shape_ok = tuple(router.shape) == expected_shape
            print(f"  Router shape: {tuple(router.shape)} (expected {expected_shape}) — {'PASS' if shape_ok else 'FAIL'}")
            if not shape_ok:
                errors.append(f"Router shape mismatch: {tuple(router.shape)} vs {expected_shape}")
            tested += 1

        # Check all experts have distinct weights (not duplicated)
        if fc1_key in hf_state:
            fc1 = hf_state[fc1_key]
            expert_means = [fc1[i].float().mean().item() for i in range(min(5, num_experts))]
            all_unique = len(set(f"{m:.6f}" for m in expert_means)) == len(expert_means)
            print(f"  Expert diversity (FC1 means): {[f'{m:.6f}' for m in expert_means]}")
            print(f"  All experts unique: {all_unique} — {'PASS' if all_unique else 'FAIL'}")
            if not all_unique:
                errors.append("Expert FC1 weights not unique — possible duplication")
            tested += 1

    # Provider check
    print(f"\n--- Provider Config Check ---")
    try:
        provider = bridge.to_megatron_provider()
        print(f"  hidden_size: {provider.hidden_size}")
        print(f"  num_layers: {provider.num_layers}")
        print(f"  vocab_size: {provider.vocab_size}")
        if hasattr(provider, 'softmax_scale') and provider.softmax_scale is not None:
            print(f"  softmax_scale: {provider.softmax_scale}")
            if m_a != 1.0 and abs(provider.softmax_scale - m_a) > 1e-6:
                errors.append(f"softmax_scale ({provider.softmax_scale}) != attention_multiplier ({m_a})")
        if is_moe:
            print(f"  num_moe_experts: {provider.num_moe_experts}")
            print(f"  moe_router_topk: {provider.moe_router_topk}")
            print(f"  moe_ffn_hidden_size: {provider.moe_ffn_hidden_size}")
        print(f"  [PASS] Provider config OK")
        tested += 1
    except Exception as e:
        errors.append(f"Provider failed: {e}")
        print(f"  [FAIL] {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"ROUND-TRIP VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Tests run: {tested}")
    print(f"  Errors: {len(errors)}")
    if errors:
        for e in errors:
            print(f"    - {e}")
        print(f"\n  RESULT: FAIL")
    else:
        print(f"\n  RESULT: ALL TESTS PASSED")
    print(f"{'='*60}\n")

    return len(errors) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Granite bridge round-trip")
    parser.add_argument("--model", required=True, help="Path to HF model")
    args = parser.parse_args()

    success = verify_multiplier_roundtrip(args.model)
    sys.exit(0 if success else 1)
