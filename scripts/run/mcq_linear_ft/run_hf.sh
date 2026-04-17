#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/scripts/run/mcq_linear_ft

EVAL_DATA_NAME=$1
CONFIG_NAME="gpt2"
MODEL_PATH="$PROJECT_BASE_PATH"/output/gpt2/HuggingFace/hf
FT_DATA="$PROJECT_BASE_PATH"/input/tokenized/gpt2/mcq_ft/"$EVAL_DATA_NAME"

echo ">>> Running tasks on gpt2-hf"
uv run bash "$SCRIPT_PATH"/run.sh \
    --config "$CONFIG_NAME" \
    --model-path "$MODEL_PATH" \
    --ft-data "$FT_DATA" \
    --evaluate-data "$EVAL_DATA_NAME"