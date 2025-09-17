#!/bin/bash
SCRIPT_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/src
DATA_NAME=wikitext-raw
CONFIG_NAME="gpt-mini"
EPOCHS=6

#           1k   5k   10k   20k   30k   40k   50k
for size in 1000 5000 10000 20000 30000 40000 50000
# for size in 1000
do
    echo "====== training on ${DATA_NAME}_$size ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train_trunc.py \
        -dp $SCRIPT_PATH/../input/${DATA_NAME} \
        -dl $size \
        -if config \
        -cn $CONFIG_NAME \
        -e $EPOCHS \
        -o $SCRIPT_PATH/../output/$CONFIG_NAME/${DATA_NAME}_${size}_${EPOCHS}
done