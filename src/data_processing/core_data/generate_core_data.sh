#! /bin/bash

# google/boolq
# train/validation
# passage question

# lighteval/squad_v2
# train/validation
# 
# data_name=trivia_qa_rc
data_name=google/boolq
for split in train validation; do
    echo ">>> processing $data_name $split"
    uv run generate_core_data.py \
        --dataset  $data_name \
        -lf hf \
        --split $split \
        --out-path $PROJECT_BASE_PATH/data/core/$data_name/$split \
        --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
        -at 10 \
        --replace-ne \
        --multi-process \
        --lower \
        --columns passage question \
        --inline-replace
done

echo ">>> Generating $data_name datasetdict"
uv run $PROJECT_BASE_PATH/scripts/tools/create_datadict.py \
    --input-path $PROJECT_BASE_PATH/data/core/$data_name \
    --output-path $PROJECT_BASE_PATH/data/core/$data_name/datadict \
    --splits train validation
