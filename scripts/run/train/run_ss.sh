#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/scripts/run/train

EVAL_DATA_NAME=$1
CONFIG_NAME="gpt2"
MODEL_PATH="$PROJECT_BASE_PATH"/output/gpt2/ss/smolLM2_135M_sents_shuffled_bs1024_ep1
EXT_TRAIN_DATA="$PROJECT_BASE_PATH"/input/tokenized/gpt2/ext/"$EVAL_DATA_NAME"_que
SFT_DATA="$PROJECT_BASE_PATH"/input/tokenized/gpt2/sft/"$EVAL_DATA_NAME"


echo ">>> Running tasks on gpt2-ss"
uv run bash "$SCRIPT_PATH"/run.sh \
    --config "$CONFIG_NAME" \
    --model-path "$MODEL_PATH" \
    --ext-train-data "$EXT_TRAIN_DATA" \
    --sft-data "$SFT_DATA" \
    --evaluate-data "$EVAL_DATA_NAME"