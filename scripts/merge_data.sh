#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling

/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/merge_dataset.py \
    -dd $BASE_PATH/data \
    -dn wikitext_with_nonce_100k,wikitext_with_nonce_100k_200k,wikitext_with_nonce_200k_300k,wikitext_with_nonce_300k_400k,wikitext_with_nonce_400k_500k \
    -o $BASE_PATH/data/wikitext_with_nonce
