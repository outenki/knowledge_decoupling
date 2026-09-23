#!/bin/bash

MODEL_CONFIG=$1
MODEL_NAME=$2
INIT_MODEL="$PROJECT_BASE_PATH/output/$MODEL_CONFIG/$MODEL_NAME"

export WANDB_MODE=offline


for SFT_DATA in \
    paws_en
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

SFT_PATH=$INIT_MODEL-sft_paws_en_train
cd $SFT_PATH
echo
echo ">>>Evaluating paws_en for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --model hf \
    --model_args pretrained=. \
    --tasks paws_en \
    --log_samples \
    --output_path eval/paws_en