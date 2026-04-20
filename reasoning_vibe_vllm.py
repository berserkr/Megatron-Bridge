#!/usr/bin/env python3
"""Fast inference with vLLM — reasoning on/off comparison.

Usage:
    # Default: 3 samples, reasoning on+off comparison, TP=4
    python eval_vllm.py \
        --model /path/to/model \
        --template chat_template.jinja \
        --data test.jsonl

    # More samples, save results
    python eval_vllm.py \
        --model /path/to/model \
        --template chat_template.jinja \
        --data test.jsonl \
        --samples 10 --output results.jsonl

    # Reasoning ON only, greedy
    python eval_vllm.py \
        --model /path/to/model \
        --template chat_template.jinja \
        --data test.jsonl \
        --mode on --greedy

    # Reasoning OFF only, custom sampling
    python eval_vllm.py \
        --model /path/to/model \
        --template chat_template.jinja \
        --data test.jsonl \
        --mode off --temperature 0.8 --top-p 0.9

    # TP=2 for smaller GPUs
    python eval_vllm.py \
        --model /path/to/model \
        --template chat_template.jinja \
        --data test.jsonl \
        --tp 2
"""

import argparse
import json
import os
import random
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_jsonl(path, limit=None):
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
            if limit and len(records) >= limit:
                break
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
        answer = "[thinking truncated — increase --max-tokens]"
    elif think_start == -1:
        answer = text.strip()

    return thinking, answer


def print_result(idx, question, thinking, answer, mode, thinking_limit):
    print(f"\n  [{mode.upper()}]")
    if thinking:
        truncated = f"\n    ... [{len(thinking):,} chars total]" if len(thinking) > thinking_limit else ""
        print(f"  Thinking ({len(thinking):,} chars):")
        for line in thinking[:thinking_limit].split("\n"):
            print(f"    {line}")
        if truncated:
            print(f"    {truncated}")
    elif mode == "on":
        print(f"  [no thinking produced]")

    print(f"\n  Answer:")
    for line in answer.split("\n"):
        print(f"    {line}")


def main():
    parser = argparse.ArgumentParser(description="Fast vLLM inference with reasoning on/off")
    parser.add_argument("--model", required=True, help="Path to HF model")
    parser.add_argument("--template", required=True, help="Path to chat template .jinja")
    parser.add_argument("--data", required=True, help="Path to test JSONL")
    parser.add_argument("--output", default=None, help="Save results to JSONL")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument("--mode", choices=["both", "on", "off"], default="both",
                        help="Reasoning mode: both (default), on, or off")

    # Generation params
    parser.add_argument("--max-tokens", type=int, default=16384, help="Max new tokens (default 16384)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (default 0.6)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling (default 0.95)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling (default 40)")
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Repetition penalty (default 1.15)")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (overrides sampling params)")

    # vLLM params
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size (default 4)")
    parser.add_argument("--gpu-mem", type=float, default=0.92, help="GPU memory utilization (default 0.92)")
    parser.add_argument("--max-model-len", type=int, default=32768, help="Max model context length")

    # Display
    parser.add_argument("--thinking-limit", type=int, default=3000, help="Max chars of thinking to display")

    args = parser.parse_args()

    # Load tokenizer for chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    with open(args.template) as f:
        tokenizer.chat_template = f.read()

    # Load and select samples
    records = load_jsonl(args.data)
    rng = random.Random(args.seed)
    selected = rng.sample(records, min(args.samples, len(records)))

    # Build prompts
    modes_to_run = []
    if args.mode in ("both", "on"):
        modes_to_run.append("on")
    if args.mode in ("both", "off"):
        modes_to_run.append("off")

    all_prompts = []
    prompt_meta = []
    for i, record in enumerate(selected):
        messages = record.get("messages", [])
        question = get_user_question(messages)
        for mode in modes_to_run:
            enable = (mode == "on")
            prompt = build_prompt(tokenizer, messages, enable_thinking=enable)
            if prompt is None:
                continue
            all_prompts.append(prompt)
            prompt_meta.append({"idx": i, "mode": mode, "question": question})

    print(f"Model:    {args.model}")
    print(f"TP:       {args.tp}")
    print(f"Samples:  {len(selected)}")
    print(f"Modes:    {', '.join(modes_to_run)}")
    print(f"Prompts:  {len(all_prompts)} total")
    print(f"Sampling: {'greedy' if args.greedy else f'temp={args.temperature} top_p={args.top_p} top_k={args.top_k} rep={args.repetition_penalty}'}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Init vLLM
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
    )

    if args.greedy:
        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0,
        )
    else:
        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )

    # Generate all at once
    print("Generating...\n")
    t0 = time.time()
    outputs = llm.generate(all_prompts, params)
    elapsed = time.time() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"Generated {total_tokens:,} tokens in {elapsed:.1f}s ({total_tokens/elapsed:,.0f} tok/s)\n")

    # Process and display results
    results = []
    current_sample = -1

    for out_idx, output in enumerate(outputs):
        meta = prompt_meta[out_idx]
        raw_text = output.outputs[0].text
        thinking, answer = extract_response(raw_text)
        num_tokens = len(output.outputs[0].token_ids)

        result = {
            "sample": meta["idx"] + 1,
            "mode": meta["mode"],
            "question": meta["question"],
            "thinking": thinking,
            "answer": answer,
            "thinking_len": len(thinking),
            "answer_len": len(answer),
            "tokens": num_tokens,
        }
        results.append(result)

        # Print header for new sample
        if meta["idx"] != current_sample:
            current_sample = meta["idx"]
            print(f"{'#'*70}")
            print(f"  SAMPLE {meta['idx']+1}")
            print(f"{'#'*70}")
            print(f"\nQUESTION:\n{meta['question'][:1500]}")
            print(f"\n{'-'*70}")

        print_result(
            meta["idx"] + 1,
            meta["question"],
            thinking,
            answer,
            meta["mode"],
            args.thinking_limit,
        )
        print(f"  [{num_tokens:,} tokens]")
        print(f"{'-'*70}")

    # Stats
    print(f"\n{'='*70}")
    print(f"STATS")
    print(f"{'='*70}")
    print(f"Total prompts:    {len(all_prompts)}")
    print(f"Total tokens:     {total_tokens:,}")
    print(f"Throughput:       {total_tokens/elapsed:,.0f} tok/s")
    print(f"Wall time:        {elapsed:.1f}s")
    print(f"Avg tokens/resp:  {total_tokens//len(all_prompts):,}")

    for mode in modes_to_run:
        mode_results = [r for r in results if r["mode"] == mode]
        if mode_results:
            avg_think = sum(r["thinking_len"] for r in mode_results) / len(mode_results)
            avg_answer = sum(r["answer_len"] for r in mode_results) / len(mode_results)
            avg_tokens = sum(r["tokens"] for r in mode_results) / len(mode_results)
            print(f"\n  {mode.upper()}:")
            print(f"    Avg thinking:  {avg_think:,.0f} chars")
            print(f"    Avg answer:    {avg_answer:,.0f} chars")
            print(f"    Avg tokens:    {avg_tokens:,.0f}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
