#MODEL=/mnt/vast/proj/checkpoints/bathen/models/exports/granite8b_sft_8k_inst-math_8k
#MODEL=/mnt/vast/proj/checkpoints/bathen/models/exports/granite30b_sft_128k
#MODEL=/mnt/vast/proj/checkpoints/bathen/models/exports/granite30b_sft_128k_special
#MODEL=/mnt/vast/proj/checkpoints/bathen/models/exports/granite_42_8b_sft_32k_cp2_diag
#TEMPLATE=../Nemotron/chat_template_g33inst.jinja
MODEL=/mnt/vast/proj/checkpoints/bathen/models/exports/granite_42_8b_sft_32k_cp1_v2
MODEL=/mnt/vast/proj/checkpoints/bathen/models/exports/granite_42_8b_sft_32k_cp2_1ep
MODEL=/mnt/vast/proj/checkpoints/bathen/models/exports/granite_cp2_transmuted2

TEMPLATE=../Nemotron/chat_template.jinja
DATA=../Nemotron/test_math_data.jsonl
#DATA=../Nemotron/test_data.jsonl

#python reasoning_vibe.py --model $MODEL --template $TEMPLATE --data $DATA
python reasoning_vibe_vllm.py \
    --model $MODEL \
    --template $TEMPLATE \
    --data $DATA \
    --category all \
    --samples-per-category 3 \
    --output granite_cp2_transmuted2.json \
    --mode on \
    --greedy 


#    --mode on \
#    --greedy 


