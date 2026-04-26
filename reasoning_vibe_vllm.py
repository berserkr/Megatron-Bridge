#!/usr/bin/env python3
"""Fast inference with vLLM — reasoning on/off comparison across categories.

Tests reasoning behavior across three categories:
  - math:    Hard math problems (from data file) — should trigger deep reasoning
  - science: Science/analysis questions (built-in) — should trigger reasoning
  - qa:      Simple factual QA (built-in) — should NOT need much reasoning

Usage:
    # Default: 3 samples per category (math, science, qa), reasoning on+off
    python reasoning_vibe_vllm.py \
        --model /path/to/model \
        --template chat_template.jinja \
        --data test.jsonl

    # Math only, 5 samples
    python reasoning_vibe_vllm.py \
        --model /path/to/model \
        --template chat_template.jinja \
        --data test.jsonl \
        --category math --samples-per-category 5

    # All categories, save results
    python reasoning_vibe_vllm.py \
        --model /path/to/model \
        --template chat_template.jinja \
        --data test.jsonl \
        --output results.jsonl

    # Science only, reasoning OFF only, greedy
    python reasoning_vibe_vllm.py \
        --model /path/to/model \
        --template chat_template.jinja \
        --category science --mode off --greedy
"""

import argparse
import json
import os
import random
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Built-in prompts for non-math categories
# ---------------------------------------------------------------------------

SCIENCE_PROMPTS = [
    "Explain why the sky is blue using Rayleigh scattering. Derive the relationship between wavelength and scattering intensity.",
    "A 2 kg block slides down a frictionless 30-degree incline. Calculate the acceleration and the speed after traveling 5 meters from rest.",
    "Describe the mechanism of CRISPR-Cas9 gene editing. What are the key molecular steps involved?",
    "Why does water expand when it freezes? Explain at the molecular level, referencing hydrogen bonding and crystal structure.",
    "Explain how a transformer neural network processes a sequence. What is the role of self-attention and how are attention scores computed?",
    "Derive the ideal gas law from kinetic theory. State your assumptions and show each step.",
]

QA_PROMPTS = [
    "What is the capital of Australia?",
    "List five programming languages commonly used for web development.",
    "Who wrote the novel '1984'?",
    "What is the chemical formula for table salt?",
    "Convert 100 degrees Fahrenheit to Celsius.",
    "Name three planets in our solar system that have rings.",
]


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


def builtin_to_messages(prompt_text):
    return [{"role": "user", "content": prompt_text}]


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


def extract_response(text, enable_thinking=True):
    eos = text.find("<|im_end|>")
    if eos != -1:
        text = text[:eos]

    think_start = text.find("<think>")
    think_end = text.find("</think>")

    thinking = ""
    answer = text

    if think_start != -1 and think_end != -1 and think_end > think_start:
        # Both tags present in output
        thinking = text[think_start + len("<think>"):think_end].strip()
        answer = text[think_end + len("</think>"):].strip()
    elif think_start != -1 and think_end == -1:
        # Opened <think> but never closed — truncated
        thinking = text[think_start + len("<think>"):].strip()
        answer = "[thinking truncated — increase --max-tokens]"
    elif think_start == -1 and think_end != -1 and enable_thinking:
        # No <think> in output but </think> found — the prompt already
        # included <think>\n so vLLM output starts inside the think block.
        thinking = text[:think_end].strip()
        answer = text[think_end + len("</think>"):].strip()
    elif think_start == -1 and think_end == -1 and enable_thinking:
        # No tags at all — entire output is thinking, never closed
        thinking = text.strip()
        answer = "[thinking truncated — increase --max-tokens]"
    else:
        answer = text.strip()

    return thinking, answer


def has_think_tag_leak(text):
    return "<think>" in text or "</think>" in text


def print_result(thinking, answer, mode, thinking_limit):
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

    if mode == "off" and has_think_tag_leak(answer):
        print(f"\n  *** WARNING: think tag leaked in reasoning-OFF output ***")


def main():
    parser = argparse.ArgumentParser(description="Multi-category reasoning on/off evaluation with vLLM")
    parser.add_argument("--model", required=True, help="Path to HF model")
    parser.add_argument("--template", required=True, help="Path to chat template .jinja")
    parser.add_argument("--data", default=None, help="Path to math test JSONL (required if category includes math)")
    parser.add_argument("--output", default=None, help="Save results to JSONL")
    parser.add_argument("--samples-per-category", type=int, default=3, help="Samples per category (default 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--category", choices=["all", "math", "science", "qa"], default="all",
                        help="Which category to test (default: all)")
    parser.add_argument("--mode", choices=["both", "on", "off"], default="both",
                        help="Reasoning mode: both (default), on, or off")

    parser.add_argument("--max-tokens", type=int, default=32768, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default 1.0, per Nemotron recommendation)")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")

    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--gpu-mem", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int, default=32768)

    parser.add_argument("--thinking-limit", type=int, default=3000, help="Max chars of thinking to display")

    args = parser.parse_args()

    categories_to_run = []
    if args.category == "all":
        categories_to_run = ["math", "science", "qa"]
    else:
        categories_to_run = [args.category]

    if "math" in categories_to_run and args.data is None:
        parser.error("--data is required when category includes math")

    modes_to_run = []
    if args.mode in ("both", "on"):
        modes_to_run.append("on")
    if args.mode in ("both", "off"):
        modes_to_run.append("off")

    rng = random.Random(args.seed)
    n = args.samples_per_category

    # --- Build per-category sample lists ---
    category_samples = {}

    if "math" in categories_to_run:
        records = load_jsonl(args.data)
        selected = rng.sample(records, min(n, len(records)))
        category_samples["math"] = [
            {"question": get_user_question(r["messages"]), "messages": r["messages"]}
            for r in selected
        ]

    if "science" in categories_to_run:
        picked = rng.sample(SCIENCE_PROMPTS, min(n, len(SCIENCE_PROMPTS)))
        category_samples["science"] = [
            {"question": p, "messages": builtin_to_messages(p)}
            for p in picked
        ]

    if "qa" in categories_to_run:
        picked = rng.sample(QA_PROMPTS, min(n, len(QA_PROMPTS)))
        category_samples["qa"] = [
            {"question": p, "messages": builtin_to_messages(p)}
            for p in picked
        ]

    # --- Load tokenizer ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    with open(args.template) as f:
        tokenizer.chat_template = f.read()

    # --- Build all prompts ---
    all_prompts = []
    prompt_meta = []

    for cat in categories_to_run:
        for i, sample in enumerate(category_samples[cat]):
            for mode in modes_to_run:
                prompt = build_prompt(tokenizer, sample["messages"], enable_thinking=(mode == "on"))
                if prompt is None:
                    continue
                all_prompts.append(prompt)
                prompt_meta.append({
                    "category": cat,
                    "idx": i,
                    "mode": mode,
                    "question": sample["question"],
                })

    total_samples = sum(len(v) for v in category_samples.values())
    print(f"Model:      {args.model}")
    print(f"TP:         {args.tp}")
    print(f"Categories: {', '.join(categories_to_run)}")
    print(f"Samples:    {total_samples} ({n} per category)")
    print(f"Modes:      {', '.join(modes_to_run)}")
    print(f"Prompts:    {len(all_prompts)} total")
    print(f"Sampling:   {'greedy' if args.greedy else f'temp={args.temperature} top_p={args.top_p}'}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # --- Init vLLM ---
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
        params = SamplingParams(max_tokens=args.max_tokens, temperature=0)
    else:
        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.15,
        )

    # --- Generate ---
    print("Generating...\n")
    t0 = time.time()
    outputs = llm.generate(all_prompts, params)
    elapsed = time.time() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"Generated {total_tokens:,} tokens in {elapsed:.1f}s ({total_tokens/elapsed:,.0f} tok/s)\n")

    # --- Process results ---
    results = []
    for out_idx, output in enumerate(outputs):
        meta = prompt_meta[out_idx]
        raw_text = output.outputs[0].text
        thinking, answer = extract_response(raw_text, enable_thinking=(meta["mode"] == "on"))
        num_tokens = len(output.outputs[0].token_ids)
        leak = has_think_tag_leak(answer) if meta["mode"] == "off" else False

        results.append({
            "category": meta["category"],
            "sample": meta["idx"] + 1,
            "mode": meta["mode"],
            "question": meta["question"],
            "thinking": thinking,
            "answer": answer,
            "thinking_len": len(thinking),
            "answer_len": len(answer),
            "tokens": num_tokens,
            "think_tag_leak": leak,
        })

    # --- Display grouped by category ---
    for cat in categories_to_run:
        cat_results = [r for r in results if r["category"] == cat]
        cat_label = {"math": "MATH", "science": "SCIENCE / ANALYSIS", "qa": "SIMPLE QA"}[cat]

        print(f"\n{'='*70}")
        print(f"  CATEGORY: {cat_label} ({len(category_samples[cat])} samples)")
        print(f"{'='*70}")

        current_sample = -1
        for r in cat_results:
            if r["sample"] != current_sample:
                current_sample = r["sample"]
                print(f"\n{'#'*70}")
                print(f"  SAMPLE {r['sample']}")
                print(f"{'#'*70}")
                print(f"\nQUESTION:\n{r['question'][:1500]}")
                print(f"\n{'-'*70}")

            print_result(r["thinking"], r["answer"], r["mode"], args.thinking_limit)
            print(f"  [{r['tokens']:,} tokens]")
            if r["think_tag_leak"]:
                print(f"  *** THINK TAG LEAK ***")
            print(f"{'-'*70}")

    # --- Stats ---
    print(f"\n{'='*70}")
    print(f"OVERALL STATS")
    print(f"{'='*70}")
    print(f"Total prompts:    {len(all_prompts)}")
    print(f"Total tokens:     {total_tokens:,}")
    print(f"Throughput:       {total_tokens/elapsed:,.0f} tok/s")
    print(f"Wall time:        {elapsed:.1f}s")
    print(f"Avg tokens/resp:  {total_tokens//max(len(all_prompts),1):,}")

    print(f"\n{'='*70}")
    print(f"STATS BY CATEGORY")
    print(f"{'='*70}")

    for cat in categories_to_run:
        cat_results = [r for r in results if r["category"] == cat]
        cat_label = {"math": "MATH", "science": "SCIENCE", "qa": "QA"}[cat]
        print(f"\n  {cat_label}:")

        for mode in modes_to_run:
            mode_results = [r for r in cat_results if r["mode"] == mode]
            if not mode_results:
                continue
            avg_think = sum(r["thinking_len"] for r in mode_results) / len(mode_results)
            avg_answer = sum(r["answer_len"] for r in mode_results) / len(mode_results)
            avg_tokens = sum(r["tokens"] for r in mode_results) / len(mode_results)
            print(f"    {mode.upper()}:")
            print(f"      Avg thinking:  {avg_think:,.0f} chars")
            print(f"      Avg answer:    {avg_answer:,.0f} chars")
            print(f"      Avg tokens:    {avg_tokens:,.0f}")

            if mode == "off":
                leaks = sum(1 for r in mode_results if r["think_tag_leak"])
                print(f"      Think-tag leaks: {leaks}/{len(mode_results)}")

    # --- Save ---
    if args.output:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
