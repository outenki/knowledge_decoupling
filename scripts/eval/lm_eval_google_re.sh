#!/bin/bash

# !!!
# !!NOTE: bad_tokens should be activated
# !!!

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

SFT_PATH=$MODEL_PATH-sft_google_re_mix_short_train
cd $SFT_PATH
echo
echo ">>>Evaluating google_re_mix_short for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --model hf \
    --model_args pretrained=. \
    --tasks google_re_mix_short \
    --log_samples \
    --output_path eval/google_re_mix_short

echo
echo ">>>Evaluating google_re_mix_conflict_short for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --model hf \
    --model_args pretrained=. \
    --tasks google_re_mix_conflict_short \
    --log_samples \
    --output_path eval/google_re_mix_conflict_short
