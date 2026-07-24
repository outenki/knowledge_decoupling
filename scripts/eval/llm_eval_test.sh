#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
MODEL_PATH=/lustre1/work/c30897/wtq/projects/knowledge_decoupling/output/openai-community/gpt2/hf_full

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

cd $MODEL_PATH

for SFT_DATA in squadv2_core; do
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

 
done
