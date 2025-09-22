#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
SCRIPT_PATH=$BASE_PATH/src
DATA_NAME=wikimedia-bs1024
CONFIG_NAME="gpt-large"
EPOCHS=1
# PRE_MODEL=$BASE_PATH/output/0830/gpt-large/wikimedia-bs1024-ep1
# CHECKPOINT=/home/pj25000107/ku50001566/projects/knowledge_decoupling/output/0830/gpt-large/checkpoints/checkpoint-wikimedia-bs1024-ep1/checkpoint-34922

echo "====== training on ${DATA_NAME}_full ======"
echo "Using checkpoint: $CHECKPOINT"
echo start time: $(date +"%D %T")
/home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
    -dp $SCRIPT_PATH/../input/${DATA_NAME} \
    -if config \
    -cn $CONFIG_NAME \
    -e $EPOCHS \
    -o $SCRIPT_PATH/../output/$CONFIG_NAME/${DATA_NAME}-ep${EPOCHS}
echo end time: $(date +"%D %T")
