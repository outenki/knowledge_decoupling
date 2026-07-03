#!/bin/bash
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

cd $MODEL_PATH


echo "Evaluating based_squad_local QA for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks based_squad_local \
    --log_samples \
    --output_path eval/based_squad_local

echo "Evaluating race_local QA for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks race_local \
    --log_samples \
    --output_path eval/race_local

echo "Evaluating squad_v2 QA for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --tasks squadv2 \
    --log_samples \
    --output_path eval/squad_v2

echo "Evaluating boolq QA for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks boolq_local \
    --log_samples \