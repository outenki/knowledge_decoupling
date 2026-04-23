#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/scripts/tools

EVAL_DATA_NAME=$1
EVAL_DATA_FORMAT=$2
CONFIG_NAME="Qwen/Qwen3.5-0.8B"
MODEL_PATH="$PROJECT_BASE_PATH"/output/Qwen/Qwen3.5-0.8B/HuggingFace/hf
EXT_TRAIN_DATA="$PROJECT_BASE_PATH"/input/tokenized/$CONFIG_NAME/ext/$EVAL_DATA_FORMAT/"$EVAL_DATA_NAME"
SFT_DATA="$PROJECT_BASE_PATH"/input/tokenized/$CONFIG_NAME/sft/$EVAL_DATA_FORMAT/"$EVAL_DATA_NAME"


# sh evaluate_json_samples.sh --model Qwen/Qwen3.5-0.8B --model-path /home/pj24001974/ku50001571/projects/knowledge_decoupling/output/Qwen/Qwen3.5-0.8B/HuggingFace/hf
echo ">>> Running tasks on hf"
uv run bash "$SCRIPT_PATH"/evaluate_json_samples.sh \
    --config "$CONFIG_NAME" \
    --model-path "$MODEL_PATH" \
    --ext-train-data "$EXT_TRAIN_DATA" \
    --sft-data "$SFT_DATA" \
    --learning-rate 2e-5 \
    --output-suffix $EVAL_DATA_FORMAT \
    --evaluate-data "$EVAL_DATA_NAME" \
    --evaluate-data-format "$EVAL_DATA_FORMAT"