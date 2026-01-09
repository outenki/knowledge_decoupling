#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
SCRIPT_PATH=$BASE_PATH/src
OUT_PATH=$BASE_PATH/output
DATA_PATH=$BASE_PATH/input/tokenized
CONFIG_NAME=$1
EPOCHS=$2


echo "====== training ${CONFIG_NAME} ======"
start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"


    # -dp $DATA_PATH/$CONFIG_NAME/train/mix_qa_test_without_options \
    # -dl 100 \
/home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
    --speedup \
    -dp $DATA_PATH/$CONFIG_NAME/train/smolLM2-bs1024 \
    -dl 100 \
    -cn $CONFIG_NAME \
    -e $EPOCHS \
    -o $OUT_PATH/$CONFIG_NAME/smolLM2/smolLM2-mix_qa_test-ep${EPOCHS}

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
