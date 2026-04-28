#!/bin/bash
TOKENIZER=$1
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
DATA_NAME="SmolLM2-20B"
DATA_PATH=$PROJECT_BASE_PATH/data/$DATA_NAME/
OUTPUT_PATH=$PROJECT_BASE_PATH/data/$DATA_NAME/tokenized/$TOKENIZER
ITER_NUM=1
SIZE=2000000
# SIZE=0
START=$(($2 * $SIZE * $ITER_NUM))
END=$(($(($2 + 1)) * $SIZE * $ITER_NUM -1))
BLOCK_SIZE=4096

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

for i in $(seq $START $SIZE $END)
do
    part=$(($i / $SIZE))
    echo
    echo "====== tokenizing part$part (from $i) ======"
    uv run python $PROJECT_BASE_PATH/src/data_processing/tokenize_and_slice_data.py \
        --tokenizer $TOKENIZER \
        -dn $DATA_NAME \
        -lf hf \
        -dc text \
        -ds train \
        -ki $DATA_PATH/kept_indices.json \
        --start-from $i \
        --limit $SIZE \
        -t \
        -s \
        -bs $BLOCK_SIZE \
        -o $OUTPUT_PATH/$TOKENIZER/part_$part
done

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
