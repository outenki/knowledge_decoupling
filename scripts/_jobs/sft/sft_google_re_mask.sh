#!/bin/bash

MODEL_CONFIG=$1
MODEL_NAME=$2
INIT_MODEL="$PROJECT_BASE_PATH/output/$MODEL_CONFIG/$MODEL_NAME"

export WANDB_MODE=offline

# SFT on google_re_mix_short datasets
SFT_DATA=google_re
cd $PROJECT_BASE_PATH/src/train
echo ">>> SFT $MODEL_CONFIG/$MODEL_NAME on $SFT_DATA"
uv run python train.py --config-name sft_train \
    base.path=$PROJECT_BASE_PATH \
    model.config="$MODEL_CONFIG" \
    model.init_model="$INIT_MODEL" \
    data.name=$SFT_DATA

SFT_PATH=$INIT_MODEL-sft_${SFT_DATA}_train
cd $SFT_PATH
echo
echo ">>>Evaluating google_re_mix_short for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --model hf \
    --model_args pretrained=. \
    --tasks google_re_mix_short \
    --log_samples \
    --output_path eval/google_re_mix_short


# # SFT on google_re_mix_short datasets after extensive pretraining
EXT_DATA=google_re_test_mask-bs4096
cd $PROJECT_BASE_PATH/src/train
echo ">>> EXT training $MODEL_CONFIG/$MODEL_NAME on $EXT_DATA"
uv run python train.py --config-name ext_train \
    base.path=$PROJECT_BASE_PATH \
    model.config="$MODEL_CONFIG" \
    model.init_model="$INIT_MODEL" \
    data.name=$EXT_DATA

EXT_MODEL=$INIT_MODEL-ext_${EXT_DATA}
SFT_DATA=google_re
cd $PROJECT_BASE_PATH/src/train
echo ">>> SFT $MODEL_CONFIG/$EXT_MODEL on $SFT_DATA"
uv run python train.py --config-name sft_train \
    base.path=$PROJECT_BASE_PATH \
    model.config="$MODEL_CONFIG" \
    model.init_model="$EXT_MODEL" \
    data.name=$SFT_DATA

SFT_PATH=$EXT_MODEL-sft_${SFT_DATA}_train
echo
echo ">>>Evaluating google_re_mix_conflict_short for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --model hf \
    --model_args pretrained=. \
    --tasks google_re_mix_conflict_short \
    --log_samples \
    --output_path eval/google_re_mix_conflict_short
