#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
DATA_NAME=SmolLM2-1.7B-100B
MULTI_PROC=10
SIZE=100000
START=$(($1 * $SIZE * $MULTI_PROC))
END=$(($(($1 + 1)) * $SIZE * $MULTI_PROC -1))
MAX_N=8


start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

for i in $(seq $START $SIZE $END)
do
    part=$(($i / $SIZE))
    echo
    echo "====== preprocess $part ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/generate_nonce_data.py \
        -d "EleutherAI/SmolLM2-135M-10B" \
        -lf hf \
        -sf $i \
        -ss text \
        -sk source \
        -sv stack_edu infimm_webmath \
        -l $SIZE \
        -mn $MAX_N \
        -o $BASE_PATH/data/$DATA_NAME/sents/mn_$MAX_N/nonce-parts/part$part
done

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
