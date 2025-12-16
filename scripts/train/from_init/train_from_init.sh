#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
SCRIPT_PATH=$BASE_PATH/src
OUT_PATH=$BASE_PATH/output
DATA_PATH=$BASE_PATH/input/tokenized
CONFIG_NAME=$1
DATA_NAME=$2
EPOCHS=$3
DATA_LIMITE=$4
SUFFIX=${5:-""}


echo "====== training on ${DATA_NAME} ======"
start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"


/home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
    --speedup \
    -dp $DATA_PATH/${DATA_NAME} \
    -cn $CONFIG_NAME \
    -e $EPOCHS \
    -dl $DATA_LIMITE \
    -o $OUT_PATH/$CONFIG_NAME/${DATA_NAME}-dl${DATA_LIMITE}-ep${EPOCHS}-tr_${SUFFIX}

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
