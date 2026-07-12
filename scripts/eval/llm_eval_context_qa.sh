#!/bin/bash
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

cd $MODEL_PATH

for SFT_DATA in squadv2 triviaqa_rc_context boolq_local; do
    cd $MODEL_PATH
    echo 
    echo ">>> Evaluating $SFT_DATA QA for: $MODEL_PATH"
    uv run accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained=. \
        --tasks $SFT_DATA \
        --log_samples \
        --output_path eval/context_qa/$SFT_DATA

 
    SFT_PATH=$INIT_MODEL-sft_${SFT_DATA}_train
    cd $SFT_PATH
    echo 
    echo ">>> Evaluating $SFT_DATA QA for: $SFT_PATH"
    uv run accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained=. \
        --tasks $SFT_DATA \
        --log_samples \
        --output_path eval/context_qa/$SFT_DATA

    SFT_PATH=$INIT_MODEL-sft_mix_train
    cd $SFT_PATH
    echo 
    echo ">>> Evaluating $SFT_DATA QA for: $SFT_PATH"
    uv run accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained=. \
        --tasks $SFT_DATA \
        --log_samples \
        --output_path eval/context_qa/$SFT_DATA
done


# echo 
# echo ">>> Evaluating squadv2 QA for: $MODEL_PATH"
# uv run accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=. \
#     --tasks squadv2 \
#     --log_samples \
#     --output_path eval/context_qa/squadv2

# echo 
# echo ">>> Evaluating boolq_local QA for: $MODEL_PATH"
# uv run accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=. \
#     --include_path $PROJECT_BASE_PATH/config/eval_tasks \
#     --tasks boolq_local \
#     --log_samples \
#     --output_path eval/context_qa/boolq_local

# echo 
# echo ">>> Evaluating triviaqa_rc_context QA for: $MODEL_PATH"
# uv run accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=. \
#     --include_path $PROJECT_BASE_PATH/config/eval_tasks \
#     --tasks triviaqa_rc_context \
#     --log_samples \
#     --output_path eval/context_qa/triviaqa_rc_context