#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
DATA_NAME=preprocessed-wikitext-103
echo "====== wikitext-103_with_nonce_full ======"
/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/generate_nonce_data.py \
    -dp $BASE_PATH/data/$DATA_NAME \
    -lf local \
    -o $BASE_PATH/data/${DATA_NAME}-nonce
