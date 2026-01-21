#!/bin/bash

BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
NONCE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/data/SmolLM2-1.7B-100B/nonce/sents/mn_8/nonce-parts
echo "==== Merging nonce vocabularies from PARTS ${NONCE_PATH} ...====="
/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/merge_nonce_vocab.py \
    -lb $NONCE_PATH \
    -wb $NONCE_PATH \
    -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/SmolLM2-1.7B-100B/nonce/vocab