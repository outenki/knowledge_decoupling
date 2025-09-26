#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
SCRIPT_PATH=$BASE_PATH/src
DATA_NAME=wikimedia-nonce-bs1024
CONFIG_NAME="gpt-large"
EPOCHS=3
PRE_MODEL=/home/pj25000107/ku50001566/projects/knowledge_decoupling/output/gpt-large/wikimedia-nonce-bs1024-ep2
CHECKPOINT=$PRE_MODEL/checkpoint-15954

echo "====== training on ${DATA_NAME}_full ======"
echo start time: $(date +"%D %T")
/home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
    -dp $SCRIPT_PATH/../input/${DATA_NAME} \
    -if pre \
    -pm $PRE_MODEL \
    -cp $CHECKPOINT \
    -e $EPOCHS \
    -o $SCRIPT_PATH/../output/$CONFIG_NAME/${DATA_NAME}-ep${EPOCHS}
echo end time: $(date +"%D %T")
