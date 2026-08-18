#!/bin/bash

# !!!
# !!NOTE: bad_tokens should be activated
# !!!
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

# for TASK in google_boolq squadv2 triviaqa triviaqa_rc_context; do
for TASK in squadv2 ; do
    # cd $MODEL_PATH
    # echo 
    # echo ">>> Evaluating $TASK QA for: $MODEL_PATH"
    # uv run accelerate launch -m lm_eval \
    #     --model hf \
    #     --model_args pretrained=. \
    #     --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    #     --tasks $TASK \
    #     --log_samples \
    #     --output_path eval/$TASK

    # SFT_PATH=$MODEL_PATH-sft_${TASK}_train
    # cd $SFT_PATH
    # echo 
    # echo ">>> Evaluating $TASK QA for: $SFT_PATH"
    # uv run accelerate launch -m lm_eval \
    #     --model hf \
    #     --model_args pretrained=. \
    #     --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    #     --tasks $TASK \
    #     --log_samples \
    #     --output_path eval/$TASK

    SFT_PATH=$MODEL_PATH-sft_${TASK}_train
    cd $SFT_PATH
    echo 
    echo ">>> Evaluating $TASK QA for: $SFT_PATH"
    uv run accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained=. \
        --include_path $PROJECT_BASE_PATH/config/eval_tasks \
        --tasks ${TASK}_context_gain \
        --log_samples \
        --output_path eval/${TASK}_context_gain

    # SFT_PATH=$MODEL_PATH-sft_mix_train
    # cd $SFT_PATH
    # echo 
    # echo ">>> Evaluating $TASK QA for: $SFT_PATH"
    # uv run accelerate launch -m lm_eval \
    #     --model hf \
    #     --model_args pretrained=. \
    #     --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    #     --tasks $TASK \
    #     --log_samples \
    #     --output_path eval/$TASK
done
