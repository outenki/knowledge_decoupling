#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-/home/pj25000107/ku50001566/projects/knowledge_decoupling}"
SCRIPT_PATH=$PROJECT_BASE_PATH/src
OUT_PATH=$PROJECT_BASE_PATH/output/HuggingFaceTB/SmolLM2-1.7B/SmolLM2-1.7B-100B-nonce-SmolLM2-1.7B-dl0-ep1-tr_0120
DATA_PATH=$PROJECT_BASE_PATH/input/tokenized/SmolLM2-1.7B-100B-nonce-SmolLM2-1.7B

CONFIG_NAME="HuggingFaceTB/SmolLM2-1.7B"
INIT_MODEL=/home/pj25000107/ku50001566/projects/knowledge_decoupling/output/HuggingFaceTB/SmolLM2-1.7B/checkpoint/SmolLM2-1.7B-100B-nonce-SmolLM2-1.7B-dl0-ep1-tr_/checkpoint-10336
EPOCHS=1


echo "====== continue training ${INIT_MODEL} ======"
start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"


/home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
    --speedup \
    -cn $CONFIG_NAME \
    -cp $INIT_MODEL \
    -dp $DATA_PATH \
    -dl 0 \
    -e $EPOCHS \
    -o $OUT_PATH

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
