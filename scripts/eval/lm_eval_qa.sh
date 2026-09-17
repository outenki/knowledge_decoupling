#!/bin/bash

# !!!
# !!NOTE: bad_tokens should be activated
# !!!

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

cd $MODEL_PATH

echo
echo ">>>Evaluating arc_easy for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --tasks arc_easy \
    --log_samples \
    --output_path eval/arc_easy

echo
echo ">>>Evaluating arc_challenge for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --tasks arc_challenge \
    --log_samples \
    --output_path eval/arc_challenge

echo
echo ">>>Evaluating commonsense_qa for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --tasks commonsense_qa \
    --log_samples \
    --output_path eval/commonsense_qa
echo

echo ">>> Evaluating ewok for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks ewok \
    --log_samples \
    --output_path eval/ewok


echo ">>> Evaluating ewok for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks  winogrande\
    --log_samples \
    --output_path eval/winogrande
