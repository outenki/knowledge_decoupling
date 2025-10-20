#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
DATA_NAME=SmolLM2
ITER_NUM=1
SIZE=1
START=$(($1 * $SIZE * $ITER_NUM))
END=$(($(($1 + 1)) * $SIZE * $ITER_NUM -1))

for i in $(seq $START $SIZE $END)
do
    part=$(($i / $SIZE))
    echo
    echo "====== preprocess $part ======"
    echo "start at $(date)"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/generate_nonce_data.py \
        -dn "EleutherAI/SmolLM2-135M-10B" \
        -lf hf \
        -o $BASE_PATH/data/$DATA_NAME/1020/test/part$part \
        -sf $i \
        -lb $BASE_PATH/data/wikimedia-nonce/vocab/lemma_blacklist \
        -wb $BASE_PATH/data/wikimedia-nonce/vocab/nonce_word_bank.json \
        -l $SIZE
    echo "end at $(date)"
done
