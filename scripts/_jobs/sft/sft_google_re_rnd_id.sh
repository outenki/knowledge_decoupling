#!/bin/bash

MODEL_CONFIG=$1
MODEL_NAME=$2
INIT_MODEL="$PROJECT_BASE_PATH/output/$MODEL_CONFIG/$MODEL_NAME"

export WANDB_MODE=offline

SFT_DATA=google_re_rnd_id
cd $PROJECT_BASE_PATH/src/train
echo ">>> SFT $MODEL_CONFIG/$MODEL_NAME on $SFT_DATA"
uv run python train.py --config-name sft_train \
    base.path=$PROJECT_BASE_PATH \
    model.config="$MODEL_CONFIG" \
    model.init_model="$INIT_MODEL" \
    data.name=$SFT_DATA

SFT_DATA=google_re_rnd_id
SFT_PATH=$INIT_MODEL-sft_${SFT_DATA}_train
cd $SFT_PATH
TASK=google_re_rnd_id
echo
echo ">>>Evaluating $TASK for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --model hf \
    --model_args pretrained=. \
    --tasks ${TASK} \
    --log_samples \
    --output_path eval/$TASK
