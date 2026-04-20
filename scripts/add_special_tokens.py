#!/usr/bin/env python3
"""Add special tokens to a Granite model and resize embeddings.

Adds chat/reasoning special tokens (<|im_start|>, <|im_end|>, <think>, </think>),
updates EOS/PAD to <|im_end|>, and pads vocab size to a multiple of 8.

Usage:
    python add_special_tokens.py \
        --model /path/to/granite-base \
        --output /path/to/granite-with-special-tokens

The output directory contains the full model + tokenizer ready for packing and training.
"""

import argparse
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pad_vocab_to_multiple(size: int, multiple: int = 8) -> int:
    """Round up vocab size to nearest multiple."""
    return math.ceil(size / multiple) * multiple


def main(model_path: str, output_path: str, multiple: int = 8):
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Original vocab size: {tokenizer.vocab_size}")
    print(f"Original EOS: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"Original PAD: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"Original BOS: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")

    # Define special tokens to add
    special_tokens = ["<|im_start|>", "<|im_end|>"]
    additional_tokens = ["<think>", "</think>"]

    # Add special tokens (these get special=True treatment)
    num_added_special = tokenizer.add_special_tokens({
        "additional_special_tokens": special_tokens,
    })
    print(f"Added {num_added_special} special tokens: {special_tokens}")

    # Add thinking tokens as regular tokens (special=False in the vocab)
    num_added_regular = tokenizer.add_tokens(additional_tokens)
    print(f"Added {num_added_regular} regular tokens: {additional_tokens}")

    # Update EOS and PAD tokens to <|im_end|>
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|im_end|>"

    # Keep BOS and UNK as-is
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<s>"
    if tokenizer.unk_token is None:
        tokenizer.unk_token = "<unk>"

    # Update model_max_length
    tokenizer.model_max_length = 262144

    print(f"\nUpdated EOS: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"Updated PAD: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"Updated BOS: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
    print(f"Updated UNK: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")

    # Verify all tokens are in the vocab
    for tok in special_tokens + additional_tokens:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        print(f"  {tok} -> id {tok_id}")

    # Calculate padded vocab size
    new_vocab_size = len(tokenizer)
    padded_vocab_size = pad_vocab_to_multiple(new_vocab_size, multiple)
    print(f"\nNew vocab size: {new_vocab_size}")
    print(f"Padded vocab size (multiple of {multiple}): {padded_vocab_size}")

    # Load model
    print(f"\nLoading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Resize embeddings to padded vocab size
    old_embed_size = model.get_input_embeddings().weight.shape[0]
    print(f"Old embedding size: {old_embed_size}")

    model.resize_token_embeddings(padded_vocab_size)

    new_embed_size = model.get_input_embeddings().weight.shape[0]
    print(f"New embedding size: {new_embed_size}")

    # Initialize new token embeddings with mean of existing embeddings
    # (better than random for faster convergence)
    with torch.no_grad():
        embed_weight = model.get_input_embeddings().weight
        mean_embed = embed_weight[:old_embed_size].mean(dim=0)
        for i in range(old_embed_size, new_embed_size):
            embed_weight[i] = mean_embed

        # If model has separate output embeddings (lm_head), initialize those too
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and output_embeddings.weight.shape[0] == new_embed_size:
            out_weight = output_embeddings.weight
            mean_out = out_weight[:old_embed_size].mean(dim=0)
            for i in range(old_embed_size, new_embed_size):
                out_weight[i] = mean_out

    # Untie lm_head so it gets saved as a separate tensor in safetensors.
    # This ensures the bridge loads it correctly with TP>1.
    model.config.tie_word_embeddings = False
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight.clone())

    # Update model config
    model.config.vocab_size = padded_vocab_size

    # Update model config
    model.config.vocab_size = padded_vocab_size
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    # Save
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving tokenizer to {output_dir}")
    tokenizer.save_pretrained(output_dir)

    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)

    # Verify
    print("\n--- Verification ---")
    tok_verify = AutoTokenizer.from_pretrained(output_dir)
    print(f"Vocab size: {len(tok_verify)}")
    print(f"EOS: {tok_verify.eos_token} (id: {tok_verify.eos_token_id})")
    print(f"PAD: {tok_verify.pad_token} (id: {tok_verify.pad_token_id})")
    print(f"BOS: {tok_verify.bos_token} (id: {tok_verify.bos_token_id})")
    print(f"model_max_length: {tok_verify.model_max_length}")

    test_text = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>reasoning</think>answer<|im_end|>"
    ids = tok_verify.encode(test_text)
    decoded = tok_verify.decode(ids)
    print(f"\nTest encode/decode:")
    print(f"  Input:   {test_text}")
    print(f"  Tokens:  {ids}")
    print(f"  Decoded: {decoded}")

    print(f"\nDone! Model with special tokens saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add special tokens to Granite model")
    parser.add_argument("--model", required=True, help="Path to base HF model")
    parser.add_argument("--output", required=True, help="Output path for model with special tokens")
    parser.add_argument("--multiple", type=int, default=8, help="Pad vocab to multiple of (default: 8)")
    args = parser.parse_args()
    main(args.model, args.output, args.multiple)
