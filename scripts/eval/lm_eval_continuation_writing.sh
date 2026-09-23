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
echo ">>>Evaluating hellaswag for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --model hf \
    --model_args pretrained=. \
    --tasks hellaswag \
    --log_samples \
    --output_path eval/hellaswag


echo
echo ">>>Evaluating lambada_openai for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --model hf \
    --model_args pretrained=. \
    --tasks lambada_openai \
    --log_samples \
    --output_path eval/lambada_openai

echo
echo ">>>Evaluating moe_storycloze for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --model hf \
    --model_args pretrained=. \
    --tasks moe_storycloze \
    --log_samples \
    --output_path eval/storycloze