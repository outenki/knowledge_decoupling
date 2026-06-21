#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

cd $MODEL_PATH
echo "Evaluating QA for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --tasks arc_easy,arc_challenge,$PROJECT_BASE_PATH/config/eval_tasks/commonsense_qa_norm.yaml \
    --log_samples \
    --output_path eval/qa