#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
DATA_NAME="SmolLM2-20B"
KEPT_INDICES_PATH=$PROJECT_BASE_PATH/data/$DATA_NAME/kept_indices.json 
OUTPUT_PATH=$PROJECT_BASE_PATH/data/$DATA_NAME/nonce/dataset
# SIZE=2000000
SIZE=1000000
START=$(($1 * $SIZE))
END=$(($(($1 + 1)) * $SIZE  -1))

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

part=$(($START / $SIZE))
echo
echo "====== generating nonce bank part$part (from $START to $END) ======"
uv run python $PROJECT_BASE_PATH/src/data_processing/nonce_data/generate_nonce_data.py \
    -d $DATA_NAME \
    -lf hf \
    -ki $KEPT_INDICES_PATH \
    --start-from $START \
    --limit $SIZE \
    -sp train \
    -nwb $PROJECT_BASE_PATH/data/$DATA_NAME/nonce/nonce_word_bank.lmdb \
    -o $OUTPUT_PATH/part_$part

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
