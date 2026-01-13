#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
SCRIPT_PATH=$BASE_PATH/src
OUT_PATH=$BASE_PATH/output/HuggingFaceTB/SmolLM2-1.7B/SmolLM2-1.7B-100B-nonce-SmolLM2-1.7B-dl0-ep1-tr_
DATA_PATH=$BASE_PATH/input/tokenized/ext/sml2

CONFIG_NAME="HuggingFaceTB/SmolLM2-1.7B"
INIT_MODEL=$BASE_PATH/output/HuggingFaceTB/SmolLM2-1.7B/SmolLM2-1.7B-100B-nonce-SmolLM2-1.7B-dl0-ep1-tr_/checkpoint-21964-0.7
EPOCHS=1


echo "====== continue training ${INIT_MODEL} ======"
start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"


/home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/train.py \
    --speedup \
    -cn $CONFIG_NAME \
    -im $INIT_MODEL \
    -dp $DATA_PATH/$CONFIG_NAME/train/mix_qa_test_without_options \
    -dl 0 \
    -dp $DATA_PATH/$CONFIG_NAME/train/smolLM2-bs1024 \
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
