#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-/home/pj25000107/ku50001566/projects/knowledge_decoupling}"
DATA_NAME=preprocessed-wikimedia
ITER_NUM=1
SIZE=100000
START=$(($1 * $SIZE * $ITER_NUM))
END=$(($(($1 + 1)) * $SIZE * $ITER_NUM -1))

for i in $(seq $START $SIZE $END)
do
    part=$(($i / $SIZE))
    echo
    echo "====== preprocess $part ======"
    echo "start at $(date)"
    /home/pj25000107/ku50001566/.local/bin/uv run python $PROJECT_BASE_PATH/src/generate_nonce_data.py \
        -dn $PROJECT_BASE_PATH/data/$DATA_NAME \
        -lf local \
        -o $PROJECT_BASE_PATH/data/wikimedia-nonce/1020/part$part \
        -sf $i \
        -lb $PROJECT_BASE_PATH/data/wikimedia-nonce/vocab/lemma_blacklist \
        -wb $PROJECT_BASE_PATH/data/wikimedia-nonce/vocab/nonce_word_bank.json \
        -l $SIZE
    echo "end at $(date)"
done

# echo
# echo "====== mine ======"
# echo "start at $(date)"
# /home/pj25000107/ku50001566/.local/bin/uv run python $PROJECT_BASE_PATH/src/generate_nonce_data.py \
#     -dn $PROJECT_BASE_PATH/data/$DATA_NAME \
#     -lf local \
#     -o $PROJECT_BASE_PATH/data/${DATA_NAME}-nonce \
#     -sf 200000 \
#     -l 100000
# echo "end at $(date)"
