#!/bin/bash

# !!!
# !!NOTE: bad_tokens should be unactivated
# !!!

MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

cd $MODEL_PATH

for TASK in google_boolq_core triviaqa_rc_context_core squadv2_core; do

    SFT_PATH=$MODEL_PATH-sft_${TASK}_angle_train
    cd $SFT_PATH
    echo 
    echo ">>> Evaluating ${TASK} QA for: $SFT_PATH"
    uv run accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained=. \
        --include_path $PROJECT_BASE_PATH/config/eval_tasks \
        --tasks ${TASK} \
        --log_samples \
        --output_path eval/${TASK}_angle
done


# for TASK in google_boolq_core; do
#     cd $MODEL_PATH
#     echo 
#     echo ">>> Evaluating ${TASK} QA for: $MODEL_PATH"
#     uv run accelerate launch -m lm_eval \
#         --model hf \
#         --model_args pretrained=. \
#         --include_path $PROJECT_BASE_PATH/config/eval_tasks \
#         --tasks ${TASK} \
#         --log_samples \
#         --output_path eval/${TASK}_angle

#     SFT_PATH=$MODEL_PATH-sft_${TASK}_train
#     cd $SFT_PATH
#     echo 
#     echo ">>> Evaluating ${TASK} QA for: $SFT_PATH"
#     uv run accelerate launch -m lm_eval \
#         --model hf \
#         --model_args pretrained=. \
#         --include_path $PROJECT_BASE_PATH/config/eval_tasks \
#         --tasks ${TASK} \
#         --log_samples \
#         --output_path eval/${TASK}_angle

#     SFT_PATH=$MODEL_PATH-sft_${TASK}_angle_train
#     cd $SFT_PATH
#     echo 
#     echo ">>> Evaluating ${TASK} QA for: $SFT_PATH"
#     uv run accelerate launch -m lm_eval \
#         --model hf \
#         --model_args pretrained=. \
#         --include_path $PROJECT_BASE_PATH/config/eval_tasks \
#         --tasks ${TASK} \
#         --log_samples \
#         --output_path eval/${TASK}_angle

# done