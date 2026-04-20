#MODEL=/mnt/vast/proj/checkpoints/bathen/models/exports/granite8b_sft_8k_inst-math_8k
MODEL=/mnt/vast/proj/checkpoints/bathen/models/exports/granite30b_sft_128k
MODEL=/mnt/vast/proj/checkpoints/bathen/models/exports/granite30b_sft_128k_special
MODEL=/mnt/vast/proj/checkpoints/bathen/models/exports/granite_42_8b_sft_128k_iter_full
#TEMPLATE=../Nemotron/chat_template_g33inst.jinja
TEMPLATE=../Nemotron/chat_template.jinja
DATA=../Nemotron/test_math_data.jsonl
#DATA=../Nemotron/test_data.jsonl

python generate.py --model $MODEL --template $TEMPLATE --data $DATA

