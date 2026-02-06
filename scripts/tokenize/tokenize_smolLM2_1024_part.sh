#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-/home/pj25000107/ku50001566/projects/knowledge_decoupling}"
DATA_PATH=$PROJECT_BASE_PATH/data/SmolLM2/sents
OUTPUT_PATH=$PROJECT_BASE_PATH/data/SmolLM2/tokenized/gpt2/sents
ITER_NUM=10
SIZE=3500000
TOKENIZER=$1
START=$(($2 * $SIZE * $ITER_NUM))
END=$(($(($2 + 1)) * $SIZE * $ITER_NUM -1))

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

for i in $(seq $START $SIZE $END)
do
    part=$(($i / $SIZE))
    echo
    echo "====== tokenizing part$part $i ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $PROJECT_BASE_PATH/src/tokenize_and_slice_data.py \
        --tokenizer $TOKENIZER \
        -dn $DATA_PATH \
        -lf local \
        -dc text \
        -ds train \
        --start-from $i \
        --limit $SIZE \
        -t \
        -s \
        -bs 1024 \
        -o $OUTPUT_PATH/$TOKENIZER/part$part
done

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
