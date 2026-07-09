# !/bin/bash

    # "meta-llama/Llama-3.2-1B" \
    # "Qwen/Qwen2.5-0.5B"
for tokenizer in \
    "HuggingFaceTB/SmolLM2-135M" \
    "allenai/OLMo-2-0425-1B"
do
    echo
    echo ">>>>>> Merge dataset for $tokenizer"

    uv run python merge_dataset.py \
        -d $PROJECT_BASE_PATH/input/tokenized/$tokenizer/sft/concat/boolq/train \
        -d $PROJECT_BASE_PATH/input/tokenized/$tokenizer/sft/concat/squad_v2/train \
        -d $PROJECT_BASE_PATH/input/tokenized/$tokenizer/sft/concat/triviaqa_rc_context/train \
        -l 10000 \
        -o $PROJECT_BASE_PATH/input/tokenized/$tokenizer/sft/concat/mix/train
done
