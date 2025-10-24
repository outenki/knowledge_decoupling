#!/bin/bash
BASE_PATH/home/pj25000107/ku50001566/projects/knowledge_decoupling
DATA_NAME=SmolLM2
ITER_NUM=1
SIZE=100
START=$(($1 * $SIZE * $ITER_NUM))
END=$(($(($1 + 1)) * $SIZE * $ITER_NUM -1))

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

for i in $(seq $START $SIZE $END)
do
    part=$(($i / $SIZE))
    echo
    echo "====== preprocess $part ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/generate_nonce_data.py \
        -dn "EleutherAI/SmolLM2-135M-10B" \
        -lf hf \
        -o $BASE_PATH/data/$DATA_NAME/1020/test/part$part \
        -sf $i \
        -ss text \
        -sk source \
        -sv stack_edu infimm_webmath \
        -lb $BASE_PATH/data/wikimedia-nonce/vocab/lemma_blacklist \
        -wb $BASE_PATH/data/wikimedia-nonce/vocab/nonce_word_bank.json \
        -l $SIZE
done

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
