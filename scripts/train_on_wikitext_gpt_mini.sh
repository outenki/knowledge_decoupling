#!/bin/bash
SCRIPT_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/src
DATA_NAME=wikitext
CONFIG_NAME="gpt-mini"

#           10k   50k   100k   200k   300k   400k   500k
for size in 10000 50000 100000 200000 300000 400000 500000
do
    echo "====== training on wikitext_$size ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
        -dp $SCRIPT_PATH/../input/${DATA_NAME} \
        -if config \
        -dl $size \
        -cn $CONFIG_NAME \
        -o $SCRIPT_PATH/../output/$CONFIG_NAME/${DATA_NAME}_$size
done