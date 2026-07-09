#!/bin/bash
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

cd $MODEL_PATH


# echo "Evaluating based_squad_local QA for: $MODEL_PATH"
# uv run accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=. \
#     --include_path $PROJECT_BASE_PATH/config/eval_tasks \
#     --tasks based_squad_local \
#     --log_samples \
#     --output_path eval/context_qa/based_squad_local

# echo "Evaluating race_local QA for: $MODEL_PATH"
# uv run accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=. \
#     --include_path $PROJECT_BASE_PATH/config/eval_tasks \
#     --tasks race_local \
#     --log_samples \
#     --output_path eval/context_qa/race_local

echo 
echo ">>> Evaluating squadv2 QA for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --tasks squadv2 \
    --log_samples \
    --output_path eval/context_qa/squadv2

echo 
echo ">>> Evaluating boolq_local QA for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks boolq_local \
    --log_samples \
    --output_path eval/context_qa/boolq_local

echo 
echo ">>> Evaluating triviaqa_rc_context QA for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks triviaqa_rc_context \
    --log_samples \
    --output_path eval/context_qa/triviaqa_rc_context