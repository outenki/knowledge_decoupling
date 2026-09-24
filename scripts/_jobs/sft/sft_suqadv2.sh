#!/bin/bash

MODEL_CONFIG=$1
MODEL_NAME=$2
INIT_MODEL="$PROJECT_BASE_PATH/output/$MODEL_CONFIG/$MODEL_NAME"

export WANDB_MODE=offline


for SFT_DATA in \
    squadv2 \
    squadv2_rnd_id
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


MODEL_PATH=$INIT_MODEL
TASK=squadv2
SFT_PATH=$MODEL_PATH-sft_${TASK}_train
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

TASK=squadv2
SFT_PATH=$MODEL_PATH-sft_${TASK}_train
cd $SFT_PATH
echo 
echo ">>> Evaluating squadv2_context_gain QA for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks squadv2_context_gain \
    --log_samples \
    --output_path eval/squadv2_context_gain


TASK=squadv2_rnd_id
SFT_PATH=$MODEL_PATH-sft_${TASK}_train
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
