"""
Verify Granite model conversion by comparing logits and generation.

Two modes:
  1. Round-trip verification (default): compare original HF vs round-tripped HF
  2. Transmute-first (recommended): transmute, then round-trip for exact results

Usage:
    # Direct round-trip (will show bfloat16 precision diffs on scaled weights)
    python verify_granite_inference.py \
        --original ibm-granite/granite-3.0-8b-base \
        --compare /path/to/roundtrip/output

    # Transmute first, then verify (recommended for exact match)
    python verify_granite_inference.py \
        --original ibm-granite/granite-3.0-8b-base \
        --compare /path/to/transmuted/roundtrip/output \
        --transmuted
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compare_models(model_a, model_b, tokenizer, label_a="Model A", label_b="Model B"):
    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "In quantum mechanics, the uncertainty principle states that",
    ]

    all_passed = True
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt!r}")
        print(f"{'='*60}")

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            logits_a = model_a(**inputs).logits
            logits_b = model_b(**inputs).logits

        # Logit comparison
        diff = (logits_a.float() - logits_b.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"Max logit diff:  {max_diff:.6e}")
        print(f"Mean logit diff: {mean_diff:.6e}")

        # Top-k comparison
        top_a = logits_a[0, -1].topk(10).indices
        top_b = logits_b[0, -1].topk(10).indices
        top_match = (top_a == top_b).all().item()
        print(f"{label_a} top-10: {top_a.tolist()}")
        print(f"{label_b} top-10: {top_b.tolist()}")
        print(f"Top-10 match: {top_match}")

        if not top_match:
            all_passed = False

        # Generation comparison
        gen_a = model_a.generate(**inputs, max_new_tokens=50, do_sample=False)
        gen_b = model_b.generate(**inputs, max_new_tokens=50, do_sample=False)

        text_a = tokenizer.decode(gen_a[0], skip_special_tokens=True)
        text_b = tokenizer.decode(gen_b[0], skip_special_tokens=True)

        gen_match = text_a == text_b
        print(f"{label_a}:  {text_a}")
        print(f"{label_b}: {text_b}")
        print(f"Generation match: {gen_match}")

        if not gen_match:
            all_passed = False

    return all_passed


def main(original_path: str, compare_path: str, transmuted: bool) -> None:
    tokenizer = AutoTokenizer.from_pretrained(original_path)

    print(f"Loading original model from {original_path} ...")
    model_orig = AutoModelForCausalLM.from_pretrained(
        original_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    print(f"Loading comparison model from {compare_path} ...")
    model_cmp = AutoModelForCausalLM.from_pretrained(
        compare_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    if transmuted:
        label = "Transmuted RT"
    else:
        label = "Round-trip"

    passed = compare_models(model_orig, model_cmp, tokenizer, "Original", label)

    print(f"\n{'='*60}")
    if passed:
        print("PASSED: All prompts match.")
    else:
        if not transmuted:
            print("FAILED: Diffs detected. This is expected for raw (non-transmuted)")
            print("Granite checkpoints due to bfloat16 precision loss in bake/unbake.")
            print("Run transmute_granite.py first, then round-trip for exact results.")
        else:
            print("FAILED: Diffs detected on transmuted checkpoint — investigate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Granite round-trip conversion via inference")
    parser.add_argument("--original", type=str, required=True, help="Original HF model path or ID")
    parser.add_argument("--compare", type=str, required=True, help="Round-tripped (or transmuted+round-tripped) HF model path")
    parser.add_argument("--transmuted", action="store_true", help="Set if the comparison model was transmuted before round-trip")
    args = parser.parse_args()
    main(args.original, args.compare, args.transmuted)
