#!/bin/bash
set -euo pipefail

HF_CONFIG=/mnt/vast/proj/checkpoints/bathen/models/base/granite-4.1-8b-base-special
OUTPUT_BASE=/mnt/vast/proj/checkpoints/bathen/models/sft

CHECKPOINTS=(
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_10m_swe_ash_128k_8b_cp2_fullcot/iter_0000323
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_10m_balanced_ash_128k_8b_cp2_fullcot/iter_0000303
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_10m_swe_ash_128k_8b_cp2/iter_0000279
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_10m_balanced_ash_128k_8b_cp2/iter_0000265
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_7m_swe_ash_128k_8b_cp2_fullcot/iter_0000219
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_7m_balanced_ash_128k_8b_cp2_fullcot/iter_0000236
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_7m_swe_ash_128k_8b_cp2/iter_0000300
    /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite_v1_sampled_7m_balanced_ash_128k_8b_cp2/iter_0000300
)

SCRIPT_DIR=/mnt/home/bathen/src/github.com/Megatron-Bridge

for ckpt in "${CHECKPOINTS[@]}"; do
    iter_name=$(basename "$ckpt")              # iter_0000300
    iter_num="${iter_name#iter_}"              # 0000300
    run_name=$(basename "$(dirname "$ckpt")")  # granite_v1_sampled_7m_balanced_ash_128k_8b_cp2
    output="${OUTPUT_BASE}/${run_name}_${iter_num}"
    echo "=============================="
    echo "Exporting: $ckpt"
    echo "Output:    $output"
    echo "=============================="
    torchrun --nproc-per-node=1 "${SCRIPT_DIR}/granite_nemo2hf.py" \
        --hf-config "$HF_CONFIG" \
        --megatron-path "$ckpt" \
        --output "$output"
    echo "Done: $iter_name"
    echo ""
done

echo "All exports complete."
