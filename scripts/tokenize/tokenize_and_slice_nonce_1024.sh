#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling

DATA_PATH=$BASE_PATH/data/wikimedia-nonce
/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/tokenize_and_slice_data.py \
    -dn $DATA_PATH/$1 -lf local -dc nonce -t -s -bs 1024 -o $DATA_PATH/$1-tokenized