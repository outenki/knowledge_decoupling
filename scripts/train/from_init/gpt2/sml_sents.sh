#!/bin/bash
PROJECT_BASE_PATH="$HOME/projects/knowledge_decoupling"
SCRIPT_PATH=$PROJECT_BASE_PATH/src
DATA_PATH=$PROJECT_BASE_PATH/input/tokenized/gpt2/train/smolLM2_135M_sents_shuffled_bs1024

CONFIG_NAME=gpt2
EPOCHS=3


echo "====== training on ${DATA_NAME} ======"
start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"


$HOME/.local/bin/uv run python $SCRIPT_PATH/train.py \
    --speedup \
    -dp $DATA_PATH \
    -cn $CONFIG_NAME \
    -e $EPOCHS \
    -dl 0 \
    -o $PROJECT_BASE_PATH/output/$CONFIG_NAME/smolLM2_135M_sents_shuffled_bs1024_ep${EPOCHS}

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
