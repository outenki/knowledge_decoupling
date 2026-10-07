#!/bin/bash

MODEL_CONFIG=$1
MODEL_NAME=$2
INIT_MODEL="$PROJECT_BASE_PATH/output/$MODEL_CONFIG/$MODEL_NAME"

export WANDB_MODE=offline

# SFT on google_re_mix_short datasets
SFT_DATA=google_re_mix_short
cd $PROJECT_BASE_PATH/src/train
echo ">>> SFT $MODEL_CONFIG/$MODEL_NAME on $SFT_DATA"
uv run python train.py --config-name sft_train \
    base.path=$PROJECT_BASE_PATH \
    model.config="$MODEL_CONFIG" \
    model.init_model="$INIT_MODEL" \
    data.name=$SFT_DATA


# # SFT on google_re_mix_short datasets after extensive pretraining
EXT_DATA=google_re_mix_short_test-bs4096
cd $PROJECT_BASE_PATH/src/train
echo ">>> EXT training $MODEL_CONFIG/$MODEL_NAME on $EXT_DATA"
uv run python train.py --config-name ext_train \
    base.path=$PROJECT_BASE_PATH \
    model.config="$MODEL_CONFIG" \
    model.init_model="$INIT_MODEL" \
    data.name=$EXT_DATA

EXT_MODEL=$INIT_MODEL-ext_${EXT_DATA}
SFT_DATA=google_re_mix_short
cd $PROJECT_BASE_PATH/src/train
echo ">>> SFT $MODEL_CONFIG/$EXT_MODEL on $SFT_DATA"
uv run python train.py --config-name sft_train \
    base.path=$PROJECT_BASE_PATH \
    model.config="$MODEL_CONFIG" \
    model.init_model="$EXT_MODEL" \
    data.name=$SFT_DATA