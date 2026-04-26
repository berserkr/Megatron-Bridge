#!/usr/bin/env python3
"""Full Granite SFT export: Megatron → HuggingFace (transmuted).

Clean single-process export — no subprocesses, no embedded scripts.
Calls Bridge APIs directly under torchrun.

Pipeline:
1. Load Megatron model via Bridge (all ranks, TP-aware)
2. Bridge export to HF format (QKV de-interleaving, multiplier un-baking, lm_head)
3. Transmute (re-bake multipliers into weights, config=1.0, sharded safetensors)
4. Verify (NaN/Inf, config, lm_head)

The result is a fully self-contained HF checkpoint with:
- All multipliers baked into weights (config values = 1.0)
- Separate lm_head (not tied to embed_tokens)
- attention_multiplier preserved (can't be baked — applied at runtime)
- Zero extra bf16 runtime operations → matches Megatron numerically
- Sharded safetensors for large models (30B+)

Usage:
    # Standard export (TP=4 matching training)
    torchrun --nproc-per-node=4 export_granite_full_v2.py \
        --hf-config /path/to/granite-base \
        --megatron-path /path/to/checkpoint/iter_XXXX \
        --output /path/to/hf/export

    # Single GPU export (reshards from TP=4 internally)
    torchrun --nproc-per-node=1 export_granite_full_v2.py \
        --hf-config /path/to/granite-base \
        --megatron-path /path/to/checkpoint/iter_XXXX \
        --output /path/to/hf/export

    # Then serve:
    #   vLLM:  python -m vllm.entrypoints.openai.api_server --model /path/to/hf/export --tp 1
    #   HF:    AutoModelForCausalLM.from_pretrained("/path/to/hf/export")
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from safetensors import safe_open
from safetensors.torch import save_file


def _parse_shard_size(size_str: str) -> int:
    size_str = size_str.strip().upper()
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    for suffix, multiplier in sorted(units.items(), key=lambda x: -len(x[0])):
        if size_str.endswith(suffix):
            return int(float(size_str[:-len(suffix)]) * multiplier)
    return int(size_str)


def _shard_and_save(all_tensors: dict, dst: Path, max_shard_bytes: int):
    tensor_sizes = {k: t.nelement() * t.element_size() for k, t in all_tensors.items()}
    total_bytes = sum(tensor_sizes.values())

    if total_bytes <= max_shard_bytes:
        save_file(all_tensors, str(dst / "model.safetensors"))
        print(f"    model.safetensors: {len(all_tensors)} tensors, {total_bytes / 1024**3:.1f} GB")
        return

    shards = []
    current_shard = {}
    current_size = 0

    for key in sorted(all_tensors.keys()):
        t_size = tensor_sizes[key]
        if current_shard and current_size + t_size > max_shard_bytes:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = all_tensors[key]
        current_size += t_size

    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    weight_map = {}

    for i, shard_tensors in enumerate(shards):
        shard_name = f"model-{i+1:05d}-of-{num_shards:05d}.safetensors"
        save_file(shard_tensors, str(dst / shard_name))
        shard_bytes = sum(tensor_sizes[k] for k in shard_tensors)
        print(f"    {shard_name}: {len(shard_tensors)} tensors, {shard_bytes / 1024**3:.1f} GB")
        for key in shard_tensors:
            weight_map[key] = shard_name

    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    }
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"    Index: {num_shards} shards, {total_bytes / 1024**3:.1f} GB total")


def patch_lm_head(bridge_output: str, model, tp_size: int, hf_config: str):
    """Ensure lm_head.weight exists in the Bridge export.

    Bridge sometimes drops lm_head when tie_word_embeddings=True in the
    original HF config. This extracts output_layer from the Megatron model
    (already in memory), un-bakes logits_scaling, and patches it in.
    """
    from megatron.core import parallel_state
    from transformers import AutoConfig

    export_path = Path(bridge_output)
    has_lm_head = False
    for sf_path in sorted(export_path.glob("*.safetensors")):
        with safe_open(str(sf_path), framework="pt", device="cpu") as sf:
            if "lm_head.weight" in sf.keys():
                has_lm_head = True
                break

    if has_lm_head:
        print("  lm_head.weight already present, skipping patch.", flush=True)
        return

    print("  lm_head.weight MISSING — patching from Megatron output_layer...", flush=True)

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_group = parallel_state.get_tensor_model_parallel_group()

    m = model[0]
    if hasattr(m, "module"):
        m = m.module

    full_w = None
    for n, p in m.named_parameters():
        if "output_layer.weight" in n:
            local_w = p.data.contiguous()
            if tp_size > 1:
                gathered = [torch.empty_like(local_w) for _ in range(tp_size)]
                torch.distributed.all_gather(gathered, local_w, group=tp_group)
                full_w = torch.cat(gathered, dim=0).cpu()
            else:
                full_w = local_w.cpu().clone()
            break

    if tp_rank != 0 or full_w is None:
        return

    cfg = AutoConfig.from_pretrained(hf_config, trust_remote_code=True)
    m_l = float(getattr(cfg, "logits_scaling", 1.0))
    lm_head = (full_w.float() * m_l).to(full_w.dtype) if m_l != 1.0 else full_w
    print(f"  output_layer: {full_w.shape}, un-baked *{m_l}", flush=True)

    # Find the shard containing embed_tokens and add lm_head there
    index_path = export_path / "model.safetensors.index.json"
    if index_path.exists() and index_path.stat().st_size > 10:
        with open(index_path) as f:
            index = json.load(f)
        embed_shard = index["weight_map"].get("model.embed_tokens.weight")
        index["weight_map"]["lm_head.weight"] = embed_shard
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        shard_path = export_path / embed_shard
    else:
        shard_path = None
        for sf_path in sorted(export_path.glob("*.safetensors")):
            with safe_open(str(sf_path), framework="pt", device="cpu") as sf:
                if "model.embed_tokens.weight" in sf.keys():
                    shard_path = sf_path
                    break
        if shard_path is None:
            shard_path = sorted(export_path.glob("*.safetensors"))[0]

    tensors = {}
    with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
        for key in sf.keys():
            tensors[key] = sf.get_tensor(key)
    tensors["lm_head.weight"] = lm_head
    save_file(tensors, str(shard_path))
    print(f"  Patched lm_head into {shard_path.name}", flush=True)


def transmute(bridge_output: str, final_output: str, max_shard_size: str = "5GB"):
    """Re-bake multipliers into weights, set config to 1.0, save with sharding."""
    print(f"\n{'='*60}")
    print("Transmute (bake multipliers -> config=1.0)")
    print(f"{'='*60}")

    src = Path(bridge_output)
    dst = Path(final_output)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    max_shard_bytes = _parse_shard_size(max_shard_size)
    print(f"  Max shard size: {max_shard_size} ({max_shard_bytes / 1024**3:.1f} GB)")

    with open(src / "config.json") as f:
        config = json.load(f)

    m_e = config.get("embedding_multiplier", 1.0)
    m_r = config.get("residual_multiplier", 1.0)
    m_l = config.get("logits_scaling", 1.0)
    print(f"  Baking: embed*={m_e}, o_proj*={m_r}, down_proj*={m_r}, lm_head/={m_l}")

    all_tensors = {}
    bake_count = 0
    for sf_path in sorted(src.glob("*.safetensors")):
        with safe_open(str(sf_path), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                t = sf.get_tensor(key)

                if key == "model.embed_tokens.weight" and m_e != 1.0:
                    t = (t.float() * m_e).to(t.dtype)
                    bake_count += 1
                elif key == "lm_head.weight" and m_l != 1.0:
                    t = (t.float() / m_l).to(t.dtype)
                    bake_count += 1
                elif re.search(r"self_attn\.o_proj\.weight$", key) and m_r != 1.0:
                    t = (t.float() * m_r).to(t.dtype)
                    bake_count += 1
                elif re.search(r"self_attn\.o_proj\.bias$", key) and m_r != 1.0:
                    t = (t.float() * m_r).to(t.dtype)
                    bake_count += 1
                elif re.search(r"mlp\.down_proj\.weight$", key) and m_r != 1.0:
                    t = (t.float() * m_r).to(t.dtype)
                    bake_count += 1
                elif re.search(r"mlp\.down_proj\.bias$", key) and m_r != 1.0:
                    t = (t.float() * m_r).to(t.dtype)
                    bake_count += 1

                all_tensors[key] = t
        print(f"    Read {sf_path.name}")

    print(f"  Baked {bake_count} weights, {len(all_tensors)} total tensors")
    _shard_and_save(all_tensors, dst, max_shard_bytes)

    config["embedding_multiplier"] = 1.0
    config["residual_multiplier"] = 1.0
    config["logits_scaling"] = 1.0
    config["tie_word_embeddings"] = False
    config["torch_dtype"] = "bfloat16"

    if "rope_parameters" in config and "rope_theta" not in config:
        config["rope_theta"] = config["rope_parameters"].get("rope_theta", 10000)
        config["rope_scaling"] = None

    with open(dst / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    for f in src.iterdir():
        if f.name != "config.json" and not f.name.endswith(".safetensors"):
            shutil.copy2(f, dst / f.name)

    print(f"\n  Final config:")
    print(f"    embedding_multiplier: 1.0")
    print(f"    residual_multiplier: 1.0")
    print(f"    logits_scaling: 1.0")
    print(f"    attention_multiplier: {config.get('attention_multiplier')} (runtime)")
    print(f"    tie_word_embeddings: False")


def verify(final_output: str):
    """Sanity check on the exported model."""
    print(f"\n{'='*60}")
    print("Verify")
    print(f"{'='*60}")

    dst = Path(final_output)

    with open(dst / "config.json") as f:
        config = json.load(f)

    checks = []
    checks.append(("tie_word_embeddings=False", config.get("tie_word_embeddings") == False))
    checks.append(("embedding_multiplier=1.0", config.get("embedding_multiplier") == 1.0))
    checks.append(("residual_multiplier=1.0", config.get("residual_multiplier") == 1.0))
    checks.append(("logits_scaling=1.0", config.get("logits_scaling") == 1.0))
    checks.append(("attention_multiplier present", config.get("attention_multiplier") is not None))

    has_lm_head = False
    has_embed = False
    lm_head_t = None
    embed_t = None

    for sf_path in sorted(dst.glob("*.safetensors")):
        with safe_open(str(sf_path), framework="pt", device="cpu") as sf:
            if "lm_head.weight" in sf.keys():
                has_lm_head = True
                lm_head_t = sf.get_tensor("lm_head.weight")
            if "model.embed_tokens.weight" in sf.keys():
                has_embed = True
                embed_t = sf.get_tensor("model.embed_tokens.weight")

    checks.append(("lm_head.weight exists", has_lm_head))
    checks.append(("embed_tokens.weight exists", has_embed))

    if lm_head_t is not None and embed_t is not None:
        checks.append(("lm_head != embed_tokens", not torch.equal(lm_head_t, embed_t)))

    has_bad = False
    for sf_path in sorted(dst.glob("*.safetensors")):
        with safe_open(str(sf_path), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                t = sf.get_tensor(key)
                if torch.isnan(t).any() or torch.isinf(t).any():
                    has_bad = True
                    print(f"  BAD: {key}")
    checks.append(("No NaN/Inf in weights", not has_bad))

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  All checks passed! Export ready: {final_output}")
    else:
        print(f"\n  SOME CHECKS FAILED")

    return all_pass


@torchrun_main
def main(
    hf_config: str,
    megatron_path: str,
    output: str,
    tp: int,
    max_shard_size: str,
    keep_bridge_export: bool,
):
    bridge_tmp = output + "_bridge_tmp"

    tp_size = tp or (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_initialized()
        else 0
    )

    print(f"\n{'#'*60}")
    print(f"# GRANITE FULL EXPORT (v2 — in-process)")
    print(f"# Megatron: {megatron_path}")
    print(f"# Output:   {output}")
    print(f"# TP:       {tp_size}")
    print(f"{'#'*60}")

    # --- Phase 1: Bridge export (all ranks participate) ---
    print(f"\n{'='*60}")
    print("Phase 1: Bridge export (QKV + multiplier un-baking + lm_head)")
    print(f"{'='*60}")

    bridge = AutoBridge.from_hf_pretrained(
        hf_config, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    mp = bridge.to_megatron_provider(load_weights=False)
    mp.tensor_model_parallel_size = tp_size
    mp.pipeline_model_parallel_size = 1
    mp.pipeline_dtype = torch.bfloat16
    mp.finalize()
    mp.initialize_model_parallel(seed=0)

    print(f"Rank {rank}: Loading Megatron model from {megatron_path}...", flush=True)
    model = bridge.load_megatron_model(
        megatron_path,
        mp_overrides={
            "tensor_model_parallel_size": tp_size,
            "pipeline_model_parallel_size": 1,
            "pipeline_dtype": torch.bfloat16,
        },
        wrap_with_ddp=False,
    )
    model = [m.cuda() for m in model]

    print(f"Rank {rank}: Saving HF checkpoint to {bridge_tmp}...", flush=True)
    bridge.save_hf_pretrained(
        model, bridge_tmp, show_progress=True, strict=False,
    )
    print(f"Rank {rank}: Bridge export complete.", flush=True)

    # --- Phase 1.5: Patch lm_head if Bridge dropped it (all ranks for TP gather) ---
    print(f"\n{'='*60}")
    print("Patch lm_head (if missing)")
    print(f"{'='*60}")
    patch_lm_head(bridge_tmp, model, tp_size, hf_config)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # --- Phase 2: Transmute + Verify (rank 0 only) ---
    if rank == 0:
        transmute(bridge_tmp, output, max_shard_size)

        passed = verify(output)

        if not keep_bridge_export and Path(bridge_tmp).exists():
            shutil.rmtree(bridge_tmp)
            print(f"\nCleaned up intermediate: {bridge_tmp}")

        print(f"\n{'#'*60}")
        if passed:
            print(f"# EXPORT SUCCESSFUL: {output}")
            print(f"#")
            print(f"# Serve with:")
            print(f"#   vLLM: python -m vllm.entrypoints.openai.api_server \\")
            print(f"#           --model {output} --tp 1")
            print(f"#   HF:   AutoModelForCausalLM.from_pretrained('{output}')")
        else:
            print(f"# EXPORT COMPLETED WITH WARNINGS")
        print(f"{'#'*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full Granite SFT export: Megatron -> HuggingFace (transmuted)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    torchrun --nproc-per-node=4 export_granite_full_v2.py \\
        --hf-config /path/to/granite-base \\
        --megatron-path /path/to/checkpoint/iter_XXXX \\
        --output /path/to/hf/export
        """,
    )
    parser.add_argument("--hf-config", required=True,
                        help="Path to base HF Granite model (for config + tokenizer)")
    parser.add_argument("--megatron-path", required=True,
                        help="Path to Megatron checkpoint directory (e.g. iter_0000540)")
    parser.add_argument("--output", required=True,
                        help="Output path for final HF model")
    parser.add_argument("--tp", type=int, default=0,
                        help="Tensor parallel size (default: auto from torchrun world size)")
    parser.add_argument("--max-shard-size", type=str, default="5GB",
                        help="Max safetensors shard size (default: 5GB)")
    parser.add_argument("--keep-bridge-export", action="store_true",
                        help="Keep intermediate Bridge export (default: clean up)")
    args = parser.parse_args()

    main(
        hf_config=args.hf_config,
        megatron_path=args.megatron_path,
        output=args.output,
        tp=args.tp,
        max_shard_size=args.max_shard_size,
        keep_bridge_export=args.keep_bridge_export,
    )
