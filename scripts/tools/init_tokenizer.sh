#!/bin/bash
DATA_PATH=$PROJECT_BASE_PATH/data/SmolLM2-135M-20B/core_ent

for TOKENIZER in  meta-llama/Llama-3.2-1B allenai/OLMo-2-0425-1B Qwen/Qwen2.5-0.5B
do
    echo ">>> Initializing tokenizer with $TOKENIZER"
    TOKENIZER_PATH=$DATA_PATH/tokenized/$TOKENIZER/tokenizer
    OUTPUT_PATH=$DATA_PATH/tokenized/$TOKENIZER/part_$PART
    uv run python $PROJECT_BASE_PATH/scripts/tools/init_tokenizer.py \
        --load-from $TOKENIZER \
        --add-token "<ENT>" \
        --add-token "<UNK>" \
        --output-path $TOKENIZER_PATH
done