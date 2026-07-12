#!/bin/bash
MODEL_PATH=/lustre1/work/c30897/wtq/projects/knowledge_decoupling/output/openai-community/gpt2/hf_full

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

cd $MODEL_PATH


# echo 
# echo ">>> Evaluating boolq_local QA for: $MODEL_PATH"
# uv run accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=. \
#     --include_path $PROJECT_BASE_PATH/config/eval_tasks \
#     --tasks boolq_local \
#     --log_samples \
#     --output_path eval/context_qa/boolq_local

echo 
echo ">>> Evaluating triviaqa_rc_context QA for: $MODEL_PATH"
uv run accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks triviaqa_rc_context \
    --log_samples \
    --output_path eval/context_qa/triviaqa_rc_context