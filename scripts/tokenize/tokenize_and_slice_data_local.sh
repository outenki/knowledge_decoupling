#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling

DATA_NAME=$1
DATA_PATH=$BASE_PATH/data/$DATA_NAME
DATA_COLUMN=$2
BATCH_SIZE=$3
/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/tokenize_and_slice_data.py \
    -dn $DATA_PATH -lf local -dc $DATA_COLUMN -s -bs 1024 -o $DATA_PATH-bs$BATCH_SIZE