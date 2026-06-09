#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/scripts/jobs/sft

EVAL_DATA_NAME=$1
EVAL_DATA_FORMAT=$2
CONFIG_NAME="Qwen/Qwen3.5-0.8B-Base"
MODEL_PATH="$PROJECT_BASE_PATH"/output/$CONFIG_NAME/SmolLM2-135M-20B-bk_th2-bs4096/base-ep0.5
EXT_TRAIN_DATA="$PROJECT_BASE_PATH"/input/tokenized/$CONFIG_NAME/ext/$EVAL_DATA_FORMAT/"$EVAL_DATA_NAME"
SFT_DATA="$PROJECT_BASE_PATH"/input/tokenized/$CONFIG_NAME/sft/$EVAL_DATA_FORMAT/"$EVAL_DATA_NAME"


echo ">>> Running tasks on SmolLM2-135M-20B-bk_th2-bs4096/base-ep0.5"
uv run bash "$SCRIPT_PATH"/run.sh \
    --config "$CONFIG_NAME" \
    --model-path "$MODEL_PATH" \
    --ext-train-data "$EXT_TRAIN_DATA" \
    --sft-data "$SFT_DATA" \
    --learning-rate 2e-5 \
    --output-suffix $EVAL_DATA_FORMAT \
    --evaluate-data "$EVAL_DATA_NAME" \
    --evaluate-data-format "$EVAL_DATA_FORMAT"