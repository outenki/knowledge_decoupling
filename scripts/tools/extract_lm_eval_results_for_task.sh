#!/bin/bash

MODEL_SUFFIX=$1
TASK=$2

for MODEL in meta-llama/Llama-3.2-1B Qwen/Qwen2.5-0.5B allenai/OLMo-2-0425-1B; do
    echo ">>> extracting $TASK results from $MODEL"
    MODEL_PATH=$PROJECT_BASE_PATH/output/${MODEL}/$MODEL_SUFFIX
    JSON_FILE=$(ls -tr "$MODEL_PATH/eval/$TASK/"results_*.json | tail -1)
    echo ">>> JSON_FILE: $JSON_FILE"
    uv run python extract_lm_eval_results.py $JSON_FILE $METRIC
done