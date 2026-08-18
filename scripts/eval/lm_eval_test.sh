#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"

for MODEL in  \
    "allenai/OLMo-2-0425-1B" \
    "meta-llama/Llama-3.2-1B" \
    "Qwen/Qwen2.5-0.5B"
do
    sh lm_eval_context_qa.sh $PROJECT_BASE_PATH/output/$MODEL/SmolLM2-135M-20B-core_ent-bs4096
done
