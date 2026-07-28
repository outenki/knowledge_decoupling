#!/bin/bash
OUTPUT_PATH=$PROJECT_BASE_PATH/input/evaluate_data/json
dn=squadv2_core
echo ">>> $dn"
uv run python generate_qa_data.py -dn boolq -rc -o $OUTPUT_PATH/$dn --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv -at 10


SFT_INPUT=$PROJECT_BASE_PATH/input/evaluate_data/json

for TOKENIZER in  meta-llama/Llama-3.2-1B allenai/OLMo-2-0425-1B Qwen/Qwen2.5-0.5B
do
    SFT_OUTPUT=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/sft/concat
    echo "============== $TOKENIZER ================"
    echo ">>>>>> $dn sft concat train"
    uv run python ./tokenize_dataset_from_json.py \
        -mp \
        --tokenizer $TOKENIZER \
        --input-path $SFT_INPUT/$dn/train.json \
        --output-path $SFT_OUTPUT/$dn/train
    echo

    echo ">>>>>> $dn sft concat test"
    uv run python ./tokenize_dataset_from_json.py \
        -mp \
        --tokenizer $TOKENIZER \
        --input-path $SFT_INPUT/$dn/test.json \
        --output-path $SFT_OUTPUT/$dn/test
    echo

    echo ">>>>>> $dn sft concat validation"
    uv run python ./tokenize_dataset_from_json.py \
        -mp \
        --tokenizer $TOKENIZER \
        --input-path $SFT_INPUT/$dn/validation.json \
        --output-path $SFT_OUTPUT/$dn/validation
    echo
done