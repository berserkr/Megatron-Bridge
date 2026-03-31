from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "/mnt/vast/proj/checkpoints/bathen/models/exports/nemotron-super-v3-rl"
model_path = "/mnt/vast/proj/checkpoints/bathen/models/exports/granite30b_sft_128k"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits[0, -1]  # logits for next token

top5 = logits.topk(5)
for val, idx in zip(top5.values, top5.indices):
    print(f"{tokenizer.decode(idx):20s} {val:.3f}")
