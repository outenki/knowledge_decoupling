#!/bin/bash

BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
SCRIPT_PATH=$BASE_PATH/src
OUT_PATH=$BASE_PATH/output
DATA_PATH=$BASE_PATH/input/tokenized
CONFIG_NAME=$1
DATA_NAME=$2
CHECKPOINT=$3
EPOCHS=$4
DATA_LIMITE=${5:-0}

echo "====== training on ${DATA_NAME}_full ======"
start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"


/home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
    -dp $DATA_PATH/${DATA_NAME} \
    -cn $CONFIG_NAME \
    -cp $CHECKPOINT \
    -e $EPOCHS \
    -dl $DATA_LIMITE \
    -o $OUT_PATH/$CONFIG_NAME/${DATA_NAME}-dl${DATA_LIMITE}-ep${EPOCHS}

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
