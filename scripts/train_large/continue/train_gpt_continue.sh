#!/bin/bash

BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
SCRIPT_PATH=$BASE_PATH/src
CONFIG_NAME=$1
DATA_NAME=$2
PRE_MODEL=$3
CHECKPOINT=$(find "$PRE_MODEL" -maxdepth 1 -type d -name "checkpoint-*" | sort | tail -n 1)
EPOCHS=$4

echo "====== training on ${DATA_NAME}_full ======"
start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"


/home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
    -dp $SCRIPT_PATH/../input/${DATA_NAME} \
    -if pre \
    -pm $PRE_MODEL \
    -cp $CHECKPOINT \
    -e $EPOCHS \
    -o $SCRIPT_PATH/../output/$CONFIG_NAME/${DATA_NAME}-ep${EPOCHS}

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
