#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-/home/pj25000107/ku50001566/projects/knowledge_decoupling}"
DATA_NAME=preprocessed-wikitext-103
echo "====== wikitext-103_with_nonce_full ======"
/home/pj25000107/ku50001566/.local/bin/uv run python $PROJECT_BASE_PATH/src/generate_nonce_data.py \
    -dp $PROJECT_BASE_PATH/data/$DATA_NAME \
    -lf local \
    -o $PROJECT_BASE_PATH/data/${DATA_NAME}-nonce
