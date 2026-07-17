#! /bin/bash

# google/boolq
# train/validation

for split in train validation; do
    uv run generate_core_data.py \
        --dataset google/boolq \
        -lf hf \
        --split $split \
        --out-path $PROJECT_BASE_PATH/data/core/google/boolq/$split \
        --replace-ne \
        --multi-process \
        --columns passage question answer \
        --inline-replace
done

# uv run $PROJECT_BASE_PATH/scripts/tools/create_datadict.py \
#     --input-path $PROJECT_BASE_PATH/data/core/google/boolq \
#     --output-path $PROJECT_BASE_PATH/data/core/google/boolq/datadict \
#     --splits train validation
