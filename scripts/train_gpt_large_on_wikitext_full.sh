#!/bin/bash
SCRIPT_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/src
DATA_NAME=wikitext-103-bs1024
CONFIG_NAME="gpt-large"
EPOCHS=6

/home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
    -dp $SCRIPT_PATH/../input/blocks/${DATA_NAME} \
    -if config \
    -cn $CONFIG_NAME \
    -e $EPOCHS \
    -o $SCRIPT_PATH/../output/$CONFIG_NAME/${DATA_NAME}_${EPOCHS}