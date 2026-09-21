#!/bin/bash

# !!!
# !!NOTE: bad_tokens should be activated
# !!!

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

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
echo ">>> Evaluating squadv2_context_gain QA for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks squadv2_context_gain \
    --log_samples \
    --output_path eval/$TASK


