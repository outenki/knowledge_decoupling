#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/scripts/run/eval

EVAL_DATA="google_re_no_context"
SCORE_ON="generation"


for model in \
    HuggingFace/hf \
    random/rnd \
    nonce/smolLM2_nonce_mn3_bs1024_dl0_ep1 \
    ss/smolLM2_135M_sents_shuffled_bs1024_ep1 \
    smolLM2/smolLM2_bs1024_dl0_ep1
do
    model_path="$PROJECT_BASE_PATH/output/gpt2/$model"
    sh "$SCRIPT_PATH/run_eval_qa.sh" \
        --config gpt2 \
        --model-path "$model_path" \
        --evaluate-data "$EVAL_DATA" \
        --score-on "$SCORE_ON"
done