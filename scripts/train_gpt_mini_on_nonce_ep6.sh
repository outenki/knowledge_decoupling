#!/bin/bash
SCRIPT_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/src
DATA_NAME=nonce
CONFIG_NAME="gpt-mini"
EPOCHS=6
MAX_DATA_LEN=128

#           10k   50k   100k   200k   300k   400k   500k
for size in 10000 50000 100000 200000 300000 400000 500000
# for size in 1000000 5000000 500000
do
    echo "====== training on ${DATA_NAME}_$size ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
        -dp $SCRIPT_PATH/../input/${DATA_NAME}_${MAX_DATA_LEN} \
        -dl $size \
        -if config \
        -cn $CONFIG_NAME \
        -e $EPOCHS \
        -o $SCRIPT_PATH/../output/$CONFIG_NAME/${DATA_NAME}_${size}_${EPOCHS}
done