#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/scripts/jobs/mcq_ft

EVAL_DATA_NAME=$1
CONFIG_NAME="Qwen/Qwen3.5-0.8B-Base"
MODEL_PATH="$PROJECT_BASE_PATH"/output/Qwen/Qwen3.5-0.8B/HuggingFace/hf
FT_DATA="$PROJECT_BASE_PATH"/input/tokenized/Qwen/Qwen3.5-0.8B/mcq_ft/"$EVAL_DATA_NAME"


echo ">>> Running tasks on hf model"
uv run bash "$SCRIPT_PATH"/run.sh \
    --config "$CONFIG_NAME" \
    --model-path "$MODEL_PATH" \
    --ft-data "$FT_DATA" \
    --learning-rate 1e-5 \
    --evaluate-data "$EVAL_DATA_NAME"