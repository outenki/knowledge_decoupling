#!/bin/bash

MODEL_CONFIG=$1
MODEL_NAME=SmolLM2-135M-20B-bs1024
INIT_MODEL="$PROJECT_BASE_PATH/output/$MODEL_CONFIG/$MODEL_NAME"

export WANDB_MODE=offline

# SFT
# for SFT_DATA in squadv2 triviaqa_rc_context boolq_local mix; do
for SFT_DATA in boolq_local; do
    cd $PROJECT_BASE_PATH/src/train
    echo ">>> SFT on $SFT_DATA"
    uv run python train.py --config-name sft_train \
        base.path=$PROJECT_BASE_PATH \
        model.config="$MODEL_CONFIG" \
        model.init_model="$INIT_MODEL" \
        data.name=$SFT_DATA
done

# for SFT_DATA in squadv2 triviaqa_rc_context boolq_local; do
for SFT_DATA in boolq_local; do
    MODEL_PATH=$INIT_MODEL-sft_${SFT_DATA}_train
    cd $MODEL_PATH
    echo 
    echo ">>> Evaluating $SFT_DATA QA for: $MODEL_PATH"
    uv run accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained=. \
        --include_path $PROJECT_BASE_PATH/config/eval_tasks \
        --tasks $SFT_DATA \
        --log_samples \
        --output_path eval/context_qa/$SFT_DATA

    # MODEL_PATH=$INIT_MODEL-sft_mix_train
    # cd $MODEL_PATH
    # echo ">>> Evaluating $SFT_DATA QA for: $MODEL_PATH"
    # uv run accelerate launch -m lm_eval \
    #     --model hf \
    #     --model_args pretrained=. \
    #     --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    #     --tasks $SFT_DATA \
    #     --log_samples \
    #     --output_path eval/context_qa/$SFT_DATA
done
