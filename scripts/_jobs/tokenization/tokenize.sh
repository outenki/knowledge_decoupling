#!/bin/bash
TOKENIZER=$1
PART=$2

BLOCK_SIZE=4096
COLUMN=text
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
INPUT_PATH=$PROJECT_BASE_PATH/data/SmolLM2-135M-20B/core_ent/parts/part_$PART
OUTPUT_PATH=$PROJECT_BASE_PATH/data/SmolLM2-135M-20B/core_ent/tokenized/$COLUMN/$TOKENIZER/part_$PART

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

uv run python $PROJECT_BASE_PATH/src/data_processing/tokenize_and_slice_data.py \
    --tokenizer $TOKENIZER \
    -dp $INPUT_PATH \
    -lf local \
    -dc $COLUMN \
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
