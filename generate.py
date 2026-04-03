import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_num}: {e}")
    return records


def drop_last_assistant_turn(messages):
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


def build_prompt(tokenizer, record):
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None

    input_messages = drop_last_assistant_turn(messages)
    if not input_messages:
        return None

    return tokenizer.apply_chat_template(
        input_messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def main(model_path, template_path, data_path, batch_size=8, max_new_tokens=256):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(template_path, "r", encoding="utf-8") as f:
        tokenizer.chat_template = f.read()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # or flash_attention_3 if supported
    ).cuda()

    model.eval()
    model.generation_config.cache_implementation = "static"
    #model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    records = load_jsonl(data_path)
    prompts = []

    for r in records:
        p = build_prompt(tokenizer, r)
        if p is not None:
            prompts.append(p)

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            max_length=131072,
            padding=True,
            truncation=True,
            pad_to_multiple_of=8,
        ).to("cuda")

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        for j, seq in enumerate(outputs):
            gen = seq[input_lens[j]:]
            text = tokenizer.decode(gen, skip_special_tokens=True)
            print(text.strip())
            print("X" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()

    main(args.model, args.template, args.data, args.batch_size, args.max_new_tokens)