#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"

for MODEL in  \
    "allenai/OLMo-2-0425-1B" \
    "Qwen/Qwen2.5-0.5B"
do
    sh lm_eval_context_qa.sh $PROJECT_BASE_PATH/output/$MODEL/SmolLM2-135M-20B-bs1024
    sh lm_eval_context_qa.sh $PROJECT_BASE_PATH/output/$MODEL/SmolLM2-135M-20B-core-bs1024
done
