#!/bin/bash
OUTPUT_PATH=$PROJECT_BASE_PATH/input/evaluate_data/json
dn=google_boolq_core_ent
echo ">>> $dn"
uv run python generate_qa_data.py \
    -dn boolq \
    -o $OUTPUT_PATH/$dn \
    --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv -at 10 \
    --split validation \
    --core-delimiter "<>" \
    --ent-generator "ENT" \
    --unk-generator "UNK"


SFT_INPUT=$PROJECT_BASE_PATH/input/evaluate_data/json

for TOKENIZER in  meta-llama/Llama-3.2-1B allenai/OLMo-2-0425-1B Qwen/Qwen2.5-0.5B
do
    SFT_OUTPUT=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/sft/concat
    echo "============== $TOKENIZER ================"
    echo ">>>>>> tokenizing $dn sft concat train"
    uv run python ./tokenize_dataset_from_json.py \
        -mp \
        --tokenizer $TOKENIZER \
        --max-length 4096 \
        --input-path $SFT_INPUT/$dn/train.json \
        --output-path $SFT_OUTPUT/$dn/train
    echo

    echo ">>>>>> tokenizing $dn sft concat test"
    uv run python ./tokenize_dataset_from_json.py \
        -mp \
        --tokenizer $TOKENIZER \
        --max-length 4096 \
        --input-path $SFT_INPUT/$dn/test.json \
        --output-path $SFT_OUTPUT/$dn/test
    echo

    echo ">>>>>> tokenizing $dn sft concat validation"
    uv run python ./tokenize_dataset_from_json.py \
        -mp \
        --tokenizer $TOKENIZER \
        --max-length 4096 \
        --input-path $SFT_INPUT/$dn/validation.json \
        --output-path $SFT_OUTPUT/$dn/validation
    echo
done