#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-/home/pj25000107/ku50001566/projects/knowledge_decoupling}"
SCRIPT_PATH=$PROJECT_BASE_PATH/src

CONFIG_NAME=$1
INIT_MODEL=$2
DATA_PATH=$3
OUT_PATH=$4
EPOCHS=$5


echo "====== training on ${DATA_PATH} ======"
start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"


uv run python $SCRIPT_PATH/train.py \
    --speedup \
    -cn $CONFIG_NAME \
    -im $INIT_MODEL \
    -dp $DATA_PATH \
    -e $EPOCHS \
    -dl 0 \
    -o $OUT_PATH

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
