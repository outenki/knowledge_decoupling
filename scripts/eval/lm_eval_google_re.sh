#!/bin/bash

# !!!
# !!NOTE: bad_tokens should be activated
# !!!

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

SFT_DATA=google_re
SFT_PATH=$MODEL_PATH-sft_${SFT_DATA}_train
cd $SFT_PATH
TASK=google_re
echo 
echo ">>> Evaluating ${TASK} QA for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks ${TASK} \
    --log_samples \
    --output_path eval/$TASK

TASK=google_re_conflict
echo 
echo ">>> Evaluating ${TASK} QA for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks ${TASK} \
    --log_samples \
    --output_path eval/$TASK

TASK=google_re_rnd_id
echo 
echo ">>> Evaluating ${TASK} QA for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks ${TASK} \
    --log_samples \
    --output_path eval/$TASK

SFT_DATA=google_re
EXT_DATA=google_re_test_mask-bs4096
SFT_PATH=$MODEL_PATH-ext_${EXT_DATA}-sft_${SFT_DATA}_train
cd $SFT_PATH
TASK=google_re_conflict
echo 
echo ">>> Evaluating ${TASK} QA for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks ${TASK} \
    --log_samples \
    --output_path eval/$TASK