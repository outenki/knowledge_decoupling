#!/bin/bash
SCRIPT_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/src
DATA_NAME=wikimedia-bs1024
CONFIG_NAME="gpt-large"
EPOCHS=2
PRE_EPOCHS=1
PRE_DATE=0830
OUTPUT_NAME=${DATA_NAME}_${EPOCHS}
CHECKPOINT=$SCRIPT_PATH/../output/$PRE_DATE/$CONFIG_NAME/${DATA_NAME}_${PRE_EPOCHS}/checkpoint-7977

echo "====== training on ${DATA_NAME}_full ======"
echo "Using checkpoint: $CHECKPOINT"
echo start time: $(date +"%T")
/home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
    -dp $SCRIPT_PATH/../input/${DATA_NAME} \
    -if config \
    -cn $CONFIG_NAME \
    -cp $CHECKPOINT \
    -e $((EPOCHS-PRE_EPOCHS)) \
    -o $SCRIPT_PATH/../output/$CONFIG_NAME/${DATA_NAME}_${EPOCHS}
echo end time: $(date +"%T")
