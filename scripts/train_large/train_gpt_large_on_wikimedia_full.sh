#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
SCRIPT_PATH=$BASE_PATH/src
DATA_NAME=wikimedia-bs1024
CONFIG_NAME="gpt-large"
EPOCHS=2
PRE_MODEL=/home/pj25000107/ku50001566/projects/knowledge_decoupling/output/gpt-large/wikimedia-bs1024-ep1
CHECKPOINT=$PRE_MODEL/checkpoint-34922

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
