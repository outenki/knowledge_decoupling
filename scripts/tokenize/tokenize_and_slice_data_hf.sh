#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling

DATA_NAME=$1
DATA_PATH=$BASE_PATH/input/tokenized/$DATA_NAME
DATA_COLUMN=$2
BATCH_SIZE=$3
/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/tokenize_and_slice_data.py \
    -dn $DATA_NAME -lf hf -dc $DATA_COLUMN -t -s -bs 1024 -o $DATA_PATH-tokenized-bs$BATCH_SIZE