#!/bin/bash
SCRIPT_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/src
for size in "10k" "50k" "100k"; do
    /home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/tokenize_data.py \
        -dp $SCRIPT_PATH/../data/wikitext_with_nonce_${size} \
        -lf local \
        -dc text \
        -dml 128 \
        -o ../input/wikitext_$size
    /home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/tokenize_data.py \
        -dp $SCRIPT_PATH/../data/wikitext_with_nonce_${size} \
        -lf local \
        -dc nonce \
        -dml 128 \
        -o ../input/nonce_$size
done