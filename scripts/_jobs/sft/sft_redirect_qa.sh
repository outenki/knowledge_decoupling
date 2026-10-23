#!/bin/bash

MODEL_CONFIG=$1
MODEL_NAME=$2
INIT_MODEL="$PROJECT_BASE_PATH/output/$MODEL_CONFIG/$MODEL_NAME"

export WANDB_MODE=offline


# sft
SFT_DATA=redirect_qa
cd $PROJECT_BASE_PATH/src/train
echo ">>> SFT $MODEL_CONFIG/$MODEL_NAME on $SFT_DATA"
uv run python train.py --config-name sft_train \
    base.path=$PROJECT_BASE_PATH \
    model.config="$MODEL_CONFIG" \
    model.init_model="$INIT_MODEL" \
    seed=42 \
    data.name=$SFT_DATA


TASK=redirect_qa
SFT_DATA=redirect_qa
SFT_PATH=$INIT_MODEL-sft_${SFT_DATA}_train
cd $SFT_PATH
echo 
echo ">>> Evaluating $TASK QA for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks $TASK \
    --log_samples \
    --output_path eval/$TASK
