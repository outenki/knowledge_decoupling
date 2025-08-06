#!/bin/bash
SCRIPT_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/src


echo "==== tokenizing wikitext ======"
/home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/tokenize_data.py \
    -dp $SCRIPT_PATH/../data/wikitext_with_nonce \
    -lf local \
    -dc text \
    -dml 128 \
    -o ../input/wikitext

echo "==== tokenizing nonce ======"
/home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/tokenize_data.py \
    -dp $SCRIPT_PATH/../data/wikitext_with_nonce \
    -lf local \
    -dc nonce \
    -dml 128 \
    -o ../input/nonce