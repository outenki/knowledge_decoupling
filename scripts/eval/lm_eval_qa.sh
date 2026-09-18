#!/bin/bash

# !!!
# !!NOTE: bad_tokens should be activated
# !!!

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

cd $MODEL_PATH

SFT_PATH=$MODEL_PATH-sft_arc_easy_train
echo
echo ">>>Evaluating arc_easy for: $SFT_PATH"
cd $SFT_PATH
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --tasks arc_easy \
    --log_samples \
    --output_path eval/arc_easy

SFT_PATH=$MODEL_PATH-sft_arc_challenge_train
echo
echo ">>>Evaluating arc_challenge for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --tasks arc_challenge \
    --log_samples \
    --output_path eval/arc_challenge

SFT_PATH=$MODEL_PATH-sft_commonsense_qa_train
echo
echo ">>>Evaluating commonsense_qa for: $SFT_PATH"
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


SFT_PATH=$MODEL_PATH-sft_winogrande_train
echo
echo ">>> Evaluating winogrande for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks  winogrande\
    --log_samples \
    --output_path eval/winogrande


SFT_PATH=$MODEL_PATH-sft_piqa_train
echo
echo ">>> Evaluating piqa for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks  piqa\
    --log_samples \
    --output_path eval/piqa