#!/bin/bash

MODEL_CONFIG=$1
MODEL_NAME=$2
INIT_MODEL="$PROJECT_BASE_PATH/output/$MODEL_CONFIG/$MODEL_NAME"

export WANDB_MODE=offline

# for SFT_DATA in triviaqa_rc_nocontext triviaqa_rc_context squadv2 google_boolq; do
for SFT_DATA in triviaqa_rc_nocontext; do
    # sft
    cd $PROJECT_BASE_PATH/src/train
    echo ">>> SFT $MODEL_CONFIG/$MODEL_NAME on $SFT_DATA"
    uv run python train.py --config-name sft_train \
        base.path=$PROJECT_BASE_PATH \
        model.config="$MODEL_CONFIG" \
        model.init_model="$INIT_MODEL" \
        data.name=$SFT_DATA
done