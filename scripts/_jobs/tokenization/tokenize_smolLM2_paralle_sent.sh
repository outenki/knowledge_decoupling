#!/bin/bash
TOKENIZER=$1
BLOCK_SIZE=$2
PART=$3

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
DATA_PATH=$PROJECT_BASE_PATH/data/SmolLM2-135M-20B/nonce/dataset
OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/train/SmolLM2-135M-20B-nonce-bs$BLOCK_SIZE
# SIZE=814692133
SIZE=80000000
START=$(($PART * $SIZE))
END=$(($(($PART + 1)) * $SIZE -1))

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

echo
echo "====== tokenizing part$PART (from $START to $END) ======"
uv run python $PROJECT_BASE_PATH/src/data_processing/tokenize_and_slice_data.py \
    --tokenizer $TOKENIZER \
    -dp $DATA_PATH \
    -lf local \
    -dc text\
    -sp train \
    --no-shuffle \
    --start-from $START \
    --limit $SIZE \
    -s \
    -bs $BLOCK_SIZE \
    -t \
    -o $OUTPUT_PATH/parts/part_$PART

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
