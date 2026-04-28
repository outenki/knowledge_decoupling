#!/bin/bash
start_time=$(date +"%s")
echo "start time: $(date -d @"$start_time" +"%D %T")"
module load cuda/13.2.0
export WANDB_MODE=offline

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/src/train

CONFIG_NAME=Qwen/Qwen3.5-0.8B-Base
EPOCHS=1
DATA_NAME=smolLM2-135M-20B-bs4096
LR="2e-4"
DATA_PATH="$PROJECT_BASE_PATH"/input/tokenized/$CONFIG_NAME/train/"$DATA_NAME"
OUTPUT_PATH="$PROJECT_BASE_PATH"/output/$CONFIG_NAME/"$DATA_NAME"/base-ep$EPOCHS


uv run python "$SCRIPT_PATH/train.py" \
    --speedup \
    -cn "$CONFIG_NAME" \
    --random-init \
    -dp "$DATA_PATH" \
    -dl 0 \
    --skip-eval \
    -lr "$LR" \
    -e "$EPOCHS" \
    --skip-eval \
    -o "$OUTPUT_PATH"


end_time=$(date +"%s")
echo "end time: $(date -d @"$end_time" +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
