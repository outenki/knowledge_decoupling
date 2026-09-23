#!/bin/bash

MODEL_CONFIG=$1
MODEL_NAME=$2
INIT_MODEL="$PROJECT_BASE_PATH/output/$MODEL_CONFIG/$MODEL_NAME"

export WANDB_MODE=offline


for SFT_DATA in \
    arc_easy \
    arc_challenge
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

SFT_PATH=$INIT_MODEL-sft_arc_easy_train
echo
echo ">>>Evaluating arc_easy for: $SFT_PATH"
cd $SFT_PATH
uv run accelerate launch -m lm_eval \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --model hf \
    --model_args pretrained=. \
    --tasks arc_easy \
    --log_samples \
    --output_path eval/arc_easy

SFT_PATH=$INIT_MODEL-sft_arc_challenge_train
echo
echo ">>>Evaluating arc_easy for: $SFT_PATH"
cd $SFT_PATH
uv run accelerate launch -m lm_eval \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --model hf \
    --model_args pretrained=. \
    --tasks arc_challenge \
    --log_samples \
    --output_path eval/arc_easy