#!/bin/bash
SCRIPT_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/src
DATA_NAME=wikimedia-bs512
CONFIG_NAME="gpt-medium"
EPOCHS=3

/home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
    -dp $SCRIPT_PATH/../input/${DATA_NAME} \
    -if config \
    -cn $CONFIG_NAME \
    -e $EPOCHS \
    -o $SCRIPT_PATH/../output/$CONFIG_NAME/${DATA_NAME}_${EPOCHS}