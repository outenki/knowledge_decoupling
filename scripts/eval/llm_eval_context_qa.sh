#!/bin/bash
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

cd $MODEL_PATH

for SFT_DATA in triviaqa_rc_context boolq_local squadv2; do
    cd $MODEL_PATH
    echo 
    echo ">>> Evaluating $SFT_DATA QA for: $MODEL_PATH"
    uv run accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained=. \
        --include_path $PROJECT_BASE_PATH/config/eval_tasks \
        --tasks $SFT_DATA \
        --log_samples \
        --output_path eval/context_qa/$SFT_DATA

 
    SFT_PATH=$MODEL_PATH-sft_${SFT_DATA}_train
    cd $SFT_PATH
    echo 
    echo ">>> Evaluating $SFT_DATA QA for: $SFT_PATH"
    uv run accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained=. \
        --include_path $PROJECT_BASE_PATH/config/eval_tasks \
        --tasks $SFT_DATA \
        --log_samples \
        --output_path eval/context_qa/$SFT_DATA

    SFT_PATH=$MODEL_PATH-sft_mix_train
    cd $SFT_PATH
    echo 
    echo ">>> Evaluating $SFT_DATA QA for: $SFT_PATH"
    uv run accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained=. \
        --include_path $PROJECT_BASE_PATH/config/eval_tasks \
        --tasks $SFT_DATA \
        --log_samples \
        --output_path eval/context_qa/$SFT_DATA
done
