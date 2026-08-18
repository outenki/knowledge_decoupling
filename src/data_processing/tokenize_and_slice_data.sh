#!/bin/bash

INPUT_PATH=$PROJECT_BASE_PATH/data/SmolLM2-135M-20B/core_ent_rp5/dataset
BLOCK_SIZE=4096

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

for TOKENIZER in  meta-llama/Llama-3.2-1B allenai/OLMo-2-0425-1B Qwen/Qwen2.5-0.5B
do
    OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/train/SmolLM2-135M-20B-rp5-bs$BLOCK_SIZE
    uv run python $PROJECT_BASE_PATH/src/data_processing/tokenize_and_slice_data.py \
        --tokenizer $TOKENIZER \
        -dp $INPUT_PATH \
        -lf local \
        -dc text \
        -sp train \
        -s \
        -bs $BLOCK_SIZE \
        -t \
        -o $OUTPUT_PATH

    OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/train/SmolLM2-135M-20B-core_ent_rp5-bs$BLOCK_SIZE
    uv run python $PROJECT_BASE_PATH/src/data_processing/tokenize_and_slice_data.py \
        --tokenizer $TOKENIZER \
        -dp $INPUT_PATH \
        -lf local \
        -dc core \
        -sp train \
        -s \
        -bs $BLOCK_SIZE \
        -t \
        -o $OUTPUT_PATH
done

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
