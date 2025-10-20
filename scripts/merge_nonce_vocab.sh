#!/bin/bash

BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
# for PARTS in 600-699 700-799 800-899 900-999 1000-1099 1100-1199 1200-1224; do
    NONCE_PATH=$BASE_PATH/data/wikimedia-nonce/vocab/parts
    echo "==== Merging nonce vocabularies from PARTS ${NONCE_PATH} ...====="
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/merge_nonce_vocab.py \
        -lb $NONCE_PATH \
        -wb $NONCE_PATH \
        -o ../data/wikimedia-nonce/vocab
# done