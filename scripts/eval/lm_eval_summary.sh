#!/bin/bash

# !!!
# !!NOTE: bad_tokens should be activated
# !!!

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

cd $MODEL_PATH

SFT_PATH=$MODEL_PATH-sft_cnn_dailymail_train
echo
echo ">>>Evaluating cnn_dailymail for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --tasks cnn_dailymail \
    --log_samples \
    --output_path eval/cnn_dailymail

SFT_PATH=$MODEL_PATH-sft_xsum_train
echo
echo ">>>Evaluating xsum for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --tasks xsum \
    --log_samples \
    --output_path eval/xsum

SFT_PATH=$MODEL_PATH-sft_samsum_train
echo
echo ">>>Evaluating samsum for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --tasks samsum \
    --log_samples \
    --output_path eval/samsum

SFT_PATH=$MODEL_PATH-sft_gigaword_train
echo
echo ">>>Evaluating gigaword for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --tasks gigaword \
    --log_samples \
    --output_path eval/gigaword