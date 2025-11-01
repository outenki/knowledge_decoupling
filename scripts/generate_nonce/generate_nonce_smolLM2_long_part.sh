#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
DATA_NAME=SmolLM2
ITER_NUM=10
SIZE=100000
MAX_N=1

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

part=$1
echo
echo "====== preprocess $part ======"
# python $BASE_PATH/src/generate_nonce_data.py \
/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/generate_nonce_data_long.py \
    -dn "EleutherAI/SmolLM2-135M-10B" \
    -lf hf \
    -sf $(($part * $SIZE)) \
    -sk source \
    -sv stack_edu infimm_webmath \
    -lb $BASE_PATH/data/wikimedia-nonce/vocab/lemma_blacklist \
    -wb $BASE_PATH/data/wikimedia-nonce/vocab/nonce_word_bank.pkl \
    -l $SIZE \
    -mn $MAX_N \
    -o $BASE_PATH/data/$DATA_NAME/long_text/nonce/mn_$MAX_N/part$part

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
