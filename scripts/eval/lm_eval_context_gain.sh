#!/bin/bash

# !!!
# !!NOTE: bad_tokens should be activated
# !!!

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

# cd $MODEL_PATH

# SFT_PATH=$MODEL_PATH-sft_google_boolq_train
# cd $SFT_PATH
# echo
# echo ">>>Evaluating google_boolq_context_gain for: $SFT_PATH"
# uv run accelerate launch -m lm_eval \
#     --include_path $PROJECT_BASE_PATH/config/eval_tasks \
#     --model hf \
#     --model_args pretrained=. \
#     --tasks google_boolq_context_gain \
#     --log_samples \
#     --output_path eval/google_boolq_context_gain


SFT_PATH=$MODEL_PATH-sft_squadv2_train
cd $SFT_PATH
echo 
echo ">>> Evaluating squadv2_context_gain QA for: $SFT_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks squadv2_context_gain \
    --log_samples \
    --output_path eval/squadv2_context_gain



# SFT_PATH=$MODEL_PATH-sft_triviaqa_rc_context_train
# cd $SFT_PATH
# echo
# echo ">>>Evaluating triviaqa_rc_context_context_gain for: $SFT_PATH"
# uv run accelerate launch -m lm_eval \
#     --include_path $PROJECT_BASE_PATH/config/eval_tasks \
#     --model hf \
#     --model_args pretrained=. \
#     --tasks triviaqa_rc_context_context_gain \
#     --log_samples \
#     --output_path eval/triviaqa_rc_context_context_gain