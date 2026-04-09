#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/scripts/run/train

EVAL_DATA_NAME=$1
# CONFIG_NAME="gpt2"
CONFIG_NAME="Qwen/Qwen3.5-0.8B-Base"
MODEL_PATH="$PROJECT_BASE_PATH"/output/$CONFIG_NAME/HuggingFace/hf
EXT_TRAIN_DATA="$PROJECT_BASE_PATH"/input/tokenized/$CONFIG_NAME/ext/"$EVAL_DATA_NAME"
SFT_DATA="$PROJECT_BASE_PATH"/input/tokenized/$CONFIG_NAME/sft/"$EVAL_DATA_NAME"


echo ">>> Running tasks on $CONFIG_NAME-hf"
uv run bash "$SCRIPT_PATH"/run.sh \
    --config "$CONFIG_NAME" \
    --model-path "$MODEL_PATH" \
    --ext-train-data "$EXT_TRAIN_DATA" \
    --sft-data "$SFT_DATA" \
    --evaluate-data "$EVAL_DATA_NAME"