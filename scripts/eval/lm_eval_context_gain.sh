#!/bin/bash

# !!!
# !!NOTE: bad_tokens should be activated
# !!!

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
MODEL_PATH=$1

SFT_PATH=$MODEL_PATH-sft_squadv2_train
TASK=squadv2_context_gain
cd $SFT_PATH
echo 
echo ">>> Evaluating $TASK for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks $TASK \
    --log_samples \
    --output_path eval/$TASK

SFT_PATH=$MODEL_PATH-sft_clasheval_train
TASK=clasheval_context_gain
cd $SFT_PATH
echo 
echo ">>> Evaluating $TASK for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks $TASK \
    --log_samples \
    --output_path eval/$TASK

SFT_PATH=$MODEL_PATH-sft_nq_swap_train
TASK=nq_swap_context_gain
cd $SFT_PATH
echo 
echo ">>> Evaluating $TASK for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks $TASK \
    --log_samples \
    --output_path eval/$TASK


SFT_PATH=$MODEL_PATH-sft_google_re_train
TASK=google_re_context_gain
cd $SFT_PATH
echo 
echo ">>> Evaluating $TASK for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks $TASK \
    --log_samples \
    --output_path eval/$TASK
