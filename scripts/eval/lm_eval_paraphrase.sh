#!/bin/bash

# !!!
# !!NOTE: bad_tokens should be activated
# !!!

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

cd $MODEL_PATH

SFT_PATH=$MODEL_PATH-sft_paws_en_train
echo
echo ">>>Evaluating paws_en for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --tasks paws_en \
    --log_samples \
    --output_path eval/paws_en

# SFT_PATH=$MODEL_PATH-sft_mrpc_train
# echo
# echo ">>>Evaluating mrpc for: $SFT_PATH"
# uv run accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=. \
#     --tasks mrpc \
#     --log_samples \
#     --output_path eval/mrpc
