#!/bin/bash

MODEL_CONFIG=$1
MODEL_NAME=$2
INIT_MODEL="$PROJECT_BASE_PATH/output/$MODEL_CONFIG/$MODEL_NAME"

export WANDB_MODE=offline


for SFT_DATA in \
    google_boolq \
    google_boolq_ent_id \
    squadv2 \
    squadv2_ent_id \
    triviaqa_rc_nocontext \
    triviaqa_rc_nocontext_ent_id  \
    triviaqa_rc_context \
    triviaqa_rc_context_ent_id
do
    # sft
    cd $PROJECT_BASE_PATH/src/train
    echo ">>> SFT $MODEL_CONFIG/$MODEL_NAME on $SFT_DATA"
    uv run python train.py --config-name sft_train \
        base.path=$PROJECT_BASE_PATH \
        model.config="$MODEL_CONFIG" \
        model.init_model="$INIT_MODEL" \
        data.name=$SFT_DATA
done