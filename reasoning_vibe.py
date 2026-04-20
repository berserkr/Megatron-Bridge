#!/usr/bin/env python3
"""Compare reasoning-on vs reasoning-off for SFT models.

Picks 3 samples from the data file, generates each with thinking enabled
and thinking disabled, and displays side by side.

Usage:
    python eval_reasoning.py \
        --model /path/to/exported/hf/model \
        --template /path/to/chat_template.jinja \
        --data /path/to/test.jsonl
"""

import argparse
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def get_user_question(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def build_prompt(tokenizer, messages, enable_thinking):
    if not isinstance(messages, list) or len(messages) == 0:
        return None
    if messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    if not messages:
        return None
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )


def extract_response(text):
    eos = text.find("<|im_end|>")
    if eos != -1:
        text = text[:eos]

    think_start = text.find("<think>")
    think_end = text.find("</think>")

    thinking = ""
    answer = text

    if think_start != -1 and think_end != -1 and think_end > think_start:
        thinking = text[think_start + len("<think>"):think_end].strip()
        answer = text[think_end + len("</think>"):].strip()
    elif think_start != -1 and think_end == -1:
        thinking = text[think_start + len("<think>"):].strip()
        answer = "[thinking truncated — increase --max-new-tokens]"
    elif think_start == -1:
        answer = text.strip()

    return thinking, answer


def generate(model, tokenizer, prompt, args):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    ).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.15,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_tokens = output[0][input_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(description="Compare reasoning on vs off")
    parser.add_argument("--model", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking-limit", type=int, default=2000)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.template) as f:
        tokenizer.chat_template = f.read()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).cuda().eval()

    records = load_jsonl(args.data)
    rng = random.Random(args.seed)
    selected = rng.sample(records, min(args.samples, len(records)))

    print(f"Loaded {len(records)} samples, selected {len(selected)}\n")

    for i, record in enumerate(selected):
        messages = record.get("messages", [])
        question = get_user_question(messages)

        print(f"{'#'*70}")
        print(f"  SAMPLE {i+1}")
        print(f"{'#'*70}")
        print(f"\nQUESTION:\n{question[:1500]}\n")

        # --- Reasoning ON ---
        prompt_on = build_prompt(tokenizer, messages, enable_thinking=True)
        if prompt_on is None:
            print("  [could not build prompt]\n")
            continue

        print(f"{'-'*70}")
        print("  REASONING ON")
        print(f"{'-'*70}")

        raw_on = generate(model, tokenizer, prompt_on, args)
        thinking_on, answer_on = extract_response(raw_on)

        if thinking_on:
            limit = args.thinking_limit
            truncated = f"\n... [{len(thinking_on)} chars total]" if len(thinking_on) > limit else ""
            print(f"\n  Thinking ({len(thinking_on)} chars):")
            for line in thinking_on[:limit].split("\n"):
                print(f"    {line}")
            if truncated:
                print(f"    {truncated}")

        print(f"\n  Answer:")
        for line in answer_on.split("\n"):
            print(f"    {line}")

        # --- Reasoning OFF ---
        prompt_off = build_prompt(tokenizer, messages, enable_thinking=False)

        print(f"\n{'-'*70}")
        print("  REASONING OFF")
        print(f"{'-'*70}")

        raw_off = generate(model, tokenizer, prompt_off, args)
        thinking_off, answer_off = extract_response(raw_off)

        if thinking_off:
            print(f"\n  [unexpected thinking block: {len(thinking_off)} chars]")

        print(f"\n  Answer:")
        for line in answer_off.split("\n"):
            print(f"    {line}")

        print()


if __name__ == "__main__":
    main()
