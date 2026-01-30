#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-/home/pj25000107/ku50001566/projects/knowledge_decoupling}"
SCRIPT_PATH=$PROJECT_BASE_PATH/src
OUT_PATH=$PROJECT_BASE_PATH/output
DATA_PATH=$PROJECT_BASE_PATH/input/tokenized/gpt2/ext

CONFIG_NAME="gpt2"
INIT_MODEL=$PROJECT_BASE_PATH/output/gpt2/nonce/smolLM2-nonce-bs1024-dl0-ep1
EPOCHS=3


echo "====== continue training ${INIT_MODEL} ======"
start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"


/home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
    --speedup \
    -cn $CONFIG_NAME \
    -im $INIT_MODEL \
    -dp $DATA_PATH/ext-test-nonce-ne \
    -e $EPOCHS \
    -dl 0\
    -o $OUT_PATH/$CONFIG_NAME/nonce/smolLM2-nonce-bs1024-dl0-ep1-ext_test_nonce_ne-ep$EPOCHS

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
