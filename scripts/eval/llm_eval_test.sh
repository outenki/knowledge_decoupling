#!/bin/bash
# MODEL=$1
# MODEL_PATH=/lustre1/work/c30897/wtq/projects/knowledge_decoupling/output/$MODEL
MODEL_PATH=/home/pj24001974/ku50001571/projects/knowledge_decoupling/output/gpt2/hf_full
SFT_DATA=squadv2_core

cd $MODEL_PATH


echo 
echo ">>> Evaluating $SFT_DATA QA for: $MODEL_PATH"
# uv run accelerate launch -m lm_eval \
uv run lm_eval \
    --model hf \
    --model_args pretrained=. \
    --include_path $PROJECT_BASE_PATH/config/eval_tasks \
    --tasks $SFT_DATA \
    --log_samples \
    --output_path eval/test/context_qa/$SFT_DATA
