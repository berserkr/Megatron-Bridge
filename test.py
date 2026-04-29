from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_path = "/mnt/vast/proj/checkpoints/bathen/models/exports/qwen3_8b_sft_128k_GLM-diag"
model_path = "/mnt/vast/proj/checkpoints/bathen/models/exports/granite_42_8b_sft_32k_cp2_diag"
model_path = "/mnt/vast/proj/checkpoints/bathen/models/exports/granite_42_8b_sft_32k_cp1_v2"
model_path = "/mnt/vast/proj/checkpoints/bathen/models/exports/granite_42_8b_sft_32k_cp1_v2_stage2"
model_path = "/mnt/vast/proj/checkpoints/bathen/models/sft/granite_v1_sampled_7m_balanced_ash_128k_8b_cp2"

template_path = "../Nemotron/chat_template.jinja"
#template_path = "/mnt/vast/proj/checkpoints/bathen/models/exports/granite30b_sft_128k_special/chat_template.jinja"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
with open(template_path) as f:
    tokenizer.chat_template = f.read()

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Describe the mechanism of CRISPR-Cas9 gene editing. What are the key molecular steps involved?"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=1, dtype="bfloat16")
out = llm.generate([prompt], SamplingParams(max_tokens=4096, temperature=0))

print("=== PROMPT ===")
print(prompt)

print("=== OUTPUT ===")
print(out[0].outputs[0].text)
