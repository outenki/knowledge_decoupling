#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/scripts/run

EVAL_DATA_NAME=$1
CONFIG_NAME="gpt2"
MODEL_PATH="$PROJECT_BASE_PATH"/output/gpt2/nonce/smolLM2_nonce_mn3_bs1024_dl0_ep1
EXT_TRAIN_DATA="$PROJECT_BASE_PATH"/input/tokenized/gpt2/ext/nonce
SFT_DATA="$PROJECT_BASE_PATH"/input/tokenized/gpt2/sft/"$EVAL_DATA_NAME"


echo ">>> Running tasks on gpt2-nonce"
uv run bash "$SCRIPT_PATH"/run.sh \
    --config "$CONFIG_NAME" \
    --model-path "$MODEL_PATH" \
    --ext-train-data "$EXT_TRAIN_DATA" \
    --sft-data "$SFT_DATA" \
    --evaluate-data "$EVAL_DATA_NAME"