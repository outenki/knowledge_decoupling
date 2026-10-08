#!/bin/bash

MODEL_CONFIG=$1
MODEL_NAME=$2
INIT_MODEL="$PROJECT_BASE_PATH/output/$MODEL_CONFIG/$MODEL_NAME"

export WANDB_MODE=offline

# SFT
SFT_DATA=clasheval
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
echo ">>>Evaluating $SFT_DATA for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --model hf \
    --model_args pretrained=. \
    --tasks $SFT_DATA \
    --log_samples \
    --output_path eval/$SFT_DATA

echo
echo ">>>Evaluating ${SFT_DATA}_rnd_id for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --model hf \
    --model_args pretrained=. \
    --tasks ${SFT_DATA}_rnd_id \
    --log_samples \
    --output_path eval/${SFT_DATA}_rnd_id

echo
echo ">>>Evaluating ${SFT_DATA}_conflict for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --model hf \
    --model_args pretrained=. \
    --tasks ${SFT_DATA}_conflict \
    --log_samples \
    --output_path eval/${SFT_DATA}_conflict


# SFT after extensive pretraining
EXT_DATA=clasheval_test-bs4096
cd $PROJECT_BASE_PATH/src/train
echo ">>> EXT training $MODEL_CONFIG/$MODEL_NAME on $EXT_DATA"
uv run python train.py --config-name ext_train \
    base.path=$PROJECT_BASE_PATH \
    model.config="$MODEL_CONFIG" \
    model.init_model="$INIT_MODEL" \
    data.name=$EXT_DATA

EXT_MODEL=$INIT_MODEL-ext_${EXT_DATA}
SFT_DATA=clasheval
cd $PROJECT_BASE_PATH/src/train
echo ">>> SFT $MODEL_CONFIG/$EXT_MODEL on $SFT_DATA"
uv run python train.py --config-name sft_train \
    base.path=$PROJECT_BASE_PATH \
    model.config="$MODEL_CONFIG" \
    model.init_model="$EXT_MODEL" \
    data.name=$SFT_DATA

SFT_PATH=$EXT_MODEL-sft_${SFT_DATA}_train
echo
echo ">>>Evaluating ${SFT_DATA}_conflict for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --model hf \
    --model_args pretrained=. \
    --tasks ${SFT_DATA}_conflict \
    --log_samples \
    --output_path eval/${SFT_DATA}_conflict
