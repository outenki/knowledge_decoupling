#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
SCRIPT_PATH=$BASE_PATH/src
OUT_PATH=$BASE_PATH/output
DATA_PATH=$BASE_PATH/input/tokenized/ext/sml2

CONFIG_NAME="HuggingFaceTB/SmolLM2-135M"
INIT_MODEL="HuggingFaceTB/SmolLM2-135M"
EPOCHS=3


echo "====== continue training ${INIT_MODEL} ======"
start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"


/home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
    --speedup \
    -cn $CONFIG_NAME \
    -im $INIT_MODEL \
    -dp $DATA_PATH/ext-train-sml2 \
    -dp $DATA_PATH/ext-val-sml2 \
    -dp $DATA_PATH/ext-test-sml2 \
    -e $EPOCHS \
    -dl 0\
    -o $OUT_PATH/$CONFIG_NAME/hf-ext_full-ep${EPOCHS}

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
