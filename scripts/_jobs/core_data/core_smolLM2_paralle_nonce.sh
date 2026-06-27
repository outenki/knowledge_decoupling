#!/bin/bash
PART=$1
TOKENIZER="meta-llama/Llama-3.2-1B"

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
DATA_PATH=$PROJECT_BASE_PATH/data/SmolLM2-135M-20B/nonce/dataset
SIZE=80000000
START=$(($PART * $SIZE))
END=$(($(($PART + 1)) * $SIZE -1))

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

echo
echo "====== tokenizing part$PART (from $START to $END) ======"
uv run python $PROJECT_BASE_PATH/src/data_processing/core_data/generate_core_data.py \
    -d $PROJECT_BASE_PATH/data/SmolLM2-135M-20B/nonce/dataset \
    -lf local \
    --start-from $START \
    --limit $SIZE \
    -rne \
    -aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
    -at 10 \
    -o $PROJECT_BASE_PATH/data/SmolLM2-135M-20B/core/parts/part_$PART \
    -mp

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
