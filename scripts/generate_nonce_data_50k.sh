#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
echo "====== wikitext_with_nonce_50k ======"
/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/generate_nonce_data.py \
    -dp $BASE_PATH/data/preprocessed_100k \
    -lf local \
    -l 100000 \
    -o $BASE_PATH/data/wikitext_with_nonce_50k
