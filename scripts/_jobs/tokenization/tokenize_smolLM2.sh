#!/bin/bash
TOKENIZER=$1
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
DATA_NAME="SmolLM2-20B"
DATA_PATH=$PROJECT_BASE_PATH/data/$DATA_NAME/
OUTPUT_PATH=$PROJECT_BASE_PATH/data/$DATA_NAME/tokenized/$TOKENIZER
# 128*1024 as recommended context length by Qwen3.5-0.8B
BLOCK_SIZE=131072

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

uv run python $PROJECT_BASE_PATH/src/data_processing/tokenize_and_slice_data.py \
    --tokenizer $TOKENIZER \
    -dn $DATA_NAME \
    -lf hf \
    -dc text \
    -ds train \
    -ki $DATA_PATH/kept_indices.json \
    -t \
    -s \
    -bs $BLOCK_SIZE \
    -o $OUTPUT_PATH

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
