#!/bin/bash

BASE=/mnt/vast/proj/checkpoints/bathen/models/base/30b-soft-lc-512k-lr1e-4-merged-3-7
TRANS=/mnt/vast/proj/checkpoints/bathen/models/base/30b-soft-lc-512k-lr1e-4-merged-3-7-trans
ROUND=/mnt/vast/proj/checkpoints/bathen/models/base/30b-soft-lc-512k-lr1e-4-merged-3-7-round

# transmute and convert... works with inference:
# Step 1: Transmute (bake multipliers, reset config to 1.0)
#python src/megatron/bridge/models/granite/transmute_granite.py --source $BASE --target $TRANS

# Step 2: Round-trip the transmuted checkpoint (bridge is now a no-op)
# note, we need to copy over tokenizer files, the do not make it after transmutation...
#cp $BASE/tokenizer* ${TRANS}/.
#python examples/conversion/hf_megatron_roundtrip.py --hf-model-id ${TRANS} --output-dir ${ROUND}

# Step 3: Verify (should be exact match)
#python examples/conversion/verify_granite_inference.py \
#      --original ${BASE}  \
#      --compare $ROUND/30b-soft-lc-512k-lr1e-4-merged-3-7-trans  \
#      --transmuted

# Round-trip
python examples/conversion/hf_megatron_roundtrip.py \
--hf-model-id $BASE \
--output-dir $ROUND

# Verify
python examples/conversion/verify_granite_inference.py \
--original $BASE \
--compare $ROUND/30b-soft-lc-512k-lr1e-4-merged-3-7