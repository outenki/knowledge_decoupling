#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
PARTS=$1  # e.g., 0-99

/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/merge_dataset.py \
    -dd $BASE_PATH/data/wikimedia-nonce/$PARTS \
    -o $BASE_PATH/data/wikimedia-nonce/merged-$PARTS
