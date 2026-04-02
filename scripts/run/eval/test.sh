#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/scripts/run/eval

EVAL_DATA="cflx_clasheval"
SCORE_ON="generation"


for model in \
    random/rnd
do
    model_path="$PROJECT_BASE_PATH/output/gpt2/$model"
    sh "$SCRIPT_PATH/run_eval_qa.sh" \
        --config gpt2 \
        --model-path "$model_path" \
        --evaluate-data "$EVAL_DATA" \
        --score-on "$SCORE_ON"
done
