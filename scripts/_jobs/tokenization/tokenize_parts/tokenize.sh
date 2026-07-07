#!/bin/bash
TOKENIZER=$1
PART=$2

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
INPUT_PATH=$PROJECT_BASE_PATH/data/SmolLM2-135M-20B/core/parts/part_$PART
OUTPUT_PATH=$PROJECT_BASE_PATH/data/SmolLM2-135M-20B/core/tokenized/$TOKENIZER/part_$PART
BLOCK_SIZE=1024

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

uv run python $PROJECT_BASE_PATH/src/data_processing/tokenize_and_slice_data.py \
    --tokenizer $TOKENIZER \
    -dp $INPUT_PATH \
    -lf local \
    -dc core \
    -sp train \
    -s \
    -bs $BLOCK_SIZE \
    -t \
    -o $OUTPUT_PATH

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
