#!/bin/bash
SCRIPT_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/src
DATA_NAME=nonce
for size in 10k 100k
do
    echo "====== training on nonce_$size ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
        -dp $SCRIPT_PATH/../input/${DATA_NAME}_$size \
        -if config \
        -cn gpt-mini \
        -o $SCRIPT_PATH/../output/gpt-mini/${DATA_NAME}_$size
done