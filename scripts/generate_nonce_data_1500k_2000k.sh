#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
echo "====== wikitext_with_nonce_1500k_2000k ======"
/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/generate_nonce_data.py \
    -dp $BASE_PATH/data/preprocessed_100k \
    -lf local \
    -sf 1500000 \
    -l  500000 \
    -o $BASE_PATH/data/wikitext_with_nonce_1500k_2000k
