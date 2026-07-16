#!/bin/bash
MODEL=$1
MODEL_PATH=/lustre1/work/c30897/wtq/projects/knowledge_decoupling/output/$MODEL

cd $MODEL_PATH

for SFT_DATA in boolq_local; do
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
 
done
