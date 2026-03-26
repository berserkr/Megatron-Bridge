import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    """
    Remove the final assistant message if the last message has role='assistant'.
    Otherwise return messages unchanged.
    """
    if not messages:
        return messages

    if messages[-1].get("role") == "assistant":
        return messages[:-1]

    return messages


def run_test(model_path, template_path, data_path):
    print(f"Loading tokenizer and model from: {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load custom Jinja template
    with open(template_path, "r", encoding="utf-8") as f:
        custom_template = f.read()
    tokenizer.chat_template = custom_template

    records = load_jsonl(data_path)
    print(f"Loaded {len(records)} records from {data_path}")

    print("\n" + "=" * 50)
    print("EXECUTING GENERATION FROM JSONL")
    print("=" * 50)

    for i, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list):
            print(f"\n[RECORD {i+1}] Skipping: missing or invalid 'messages' field")
            continue

        input_messages = drop_last_assistant_turn(messages)

        if not input_messages:
            print(f"\n[RECORD {i+1}] Skipping: no usable messages after trimming")
            continue

        # Get raw formatted prompt text
        raw_prompt_text = tokenizer.apply_chat_template(
            input_messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenize for generation
        inputs = tokenizer(raw_prompt_text, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

        print(f"\n[RECORD {i+1}]")
        print("[INPUT MESSAGES]:")
        for msg in input_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            print(f"- {role}: {content}")

        print("\n[RAW TEMPLATE]:")
        print(raw_prompt_text)
        print("-" * 15)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"[RESPONSE]: {response.strip()}")
        print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument(
        "--data",
        type=str,
        default="../Nemotron/test_data.jsonl",
        help="Path to JSONL file containing a 'messages' field",
    )
    args = parser.parse_args()

    run_test(args.model, args.template, args.data)