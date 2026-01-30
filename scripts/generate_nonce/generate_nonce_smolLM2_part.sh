#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-/home/pj25000107/ku50001566/projects/knowledge_decoupling}"
DATA_NAME=SmolLM2
SIZE=100000

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

for part in 5 10 26 37
do
    echo
    echo "====== preprocess $part ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $PROJECT_BASE_PATH/src/generate_nonce_data_long.py \
        -dn "EleutherAI/SmolLM2-135M-10B" \
        -lf hf \
        -o $PROJECT_BASE_PATH/data/$DATA_NAME/1020/test/part$part \
        -sf $(($part * $SIZE)) \
        -ss text \
        -sk source \
        -sv stack_edu infimm_webmath \
        -lb $PROJECT_BASE_PATH/data/wikimedia-nonce/vocab/lemma_blacklist \
        -wb $PROJECT_BASE_PATH/data/wikimedia-nonce/vocab/nonce_word_bank.json \
        -l $SIZE
done

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
