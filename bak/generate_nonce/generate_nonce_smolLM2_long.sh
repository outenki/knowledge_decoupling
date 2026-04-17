#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-/home/pj25000107/ku50001566/projects/knowledge_decoupling}"
DATA_NAME=SmolLM2
ITER_NUM=1
SIZE=10000
START=$(($1 * $SIZE * $ITER_NUM))
END=$(($(($1 + 1)) * $SIZE * $ITER_NUM -1))
MAX_N=1

start_time=$(date +"%s")

if date -d "@$start_time" >/dev/null 2>&1; then
    # GNU date (Linux)
    echo "start time: $(date -d "@$start_time" +"%m/%d/%y %T")"
else
    # BSD date (macOS)
    echo "start time: $(date -r $start_time +"%m/%d/%y %T")"
fi

for i in $(seq $START $SIZE $END)
do
    part=$(($i / $SIZE))
    echo
    echo "====== preprocess $part ======"
    # python $PROJECT_BASE_PATH/src/generate_nonce_data.py \
    /home/pj25000107/ku50001566/.local/bin/uv run python $PROJECT_BASE_PATH/src/generate_nonce_data_long.py \
        -dn "EleutherAI/SmolLM2-135M-10B" \
        -lf hf \
        -sf $i \
        -sk source \
        -sv stack_edu infimm_webmath \
        -lb $PROJECT_BASE_PATH/data/wikimedia-nonce/vocab/lemma_blacklist \
        -wb $PROJECT_BASE_PATH/data/wikimedia-nonce/vocab/nonce_word_bank.pkl \
        -l $SIZE \
        -mn $MAX_N \
        -o $PROJECT_BASE_PATH/data/$DATA_NAME/test/nonce/mn_$MAX_N/part$part
done

end_time=$(date +"%s")

if date -d "@$end_time" >/dev/null 2>&1; then
    # GNU date (Linux)
    echo "end time: $(date -d "@$end_time" +"%m/%d/%y %T")"
else
    # BSD date (macOS)
    echo "end time: $(date -r $end_time +"%m/%d/%y %T")"
fi
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
