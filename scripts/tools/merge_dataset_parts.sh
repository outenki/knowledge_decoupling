# !/bin/bash

    # "HuggingFaceTB/SmolLM2-135M" \
DATA_NAME=SmolLM2-135M-20B
PROCESS=core_ent
BLOCK_SIZE=4096
for tokenizer in \
    "allenai/OLMo-2-0425-1B" \
    "meta-llama/Llama-3.2-1B" \
    "Qwen/Qwen2.5-0.5B"
do
    echo
    echo ">>>>>> Merge dataset for $tokenizer"

    uv run python merge_dataset_parts.py \
        --data-dir $PROJECT_BASE_PATH/data/$DATA_NAME/$PROCESS/tokenized/$tokenizer \
        --part-range 0 10 \
        -o $PROJECT_BASE_PATH/input/tokenized/$tokenizer/train/$DATA_NAME-$PROCESS-bs$BLOCK_SIZE
done
