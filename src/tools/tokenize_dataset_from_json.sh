#! /bin/bash
# For extensive pretraining
INPUT_PATH=$PROJECT_BASE_PATH/data/ext
OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/gpt2/ext

for dn in \
    google_re_no_context
do
    echo
    echo ">>>>>> $dn"
    uv run python ./tokenize_dataset_from_json.py \
        --tokenizer gpt2 \
        --input-path $INPUT_PATH/$dn/train.json \
        --output-path $OUTPUT_PATH/$dn/train
    uv run python ./tokenize_dataset_from_json.py \
        --tokenizer gpt2 \
        --input-path $INPUT_PATH/$dn/test.json \
        --output-path $OUTPUT_PATH/$dn/test
    uv run python ./tokenize_dataset_from_json.py \
        -sa \
        --tokenizer gpt2 \
        --input-path $INPUT_PATH/$dn/train.json \
        --output-path $OUTPUT_PATH/"$dn"_que/train
    uv run python ./tokenize_dataset_from_json.py \
        -sa \
        --tokenizer gpt2 \
        --input-path $INPUT_PATH/$dn/test.json \
        --output-path $OUTPUT_PATH/"$dn"_que/test
done

INPUT_PATH=$PROJECT_BASE_PATH/data/ext
OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/gpt2/ext
for dn in \
    google_re_no_context
do
    echo
    echo ">>>>>> $dn"
    uv run python ./tokenize_dataset_from_json.py \
        -mp \
        --tokenizer gpt2 \
        --input-path $INPUT_PATH/$dn/train.json \
        --output-path $OUTPUT_PATH/$dn/train
    uv run python ./tokenize_dataset_from_json.py \
        -mp \
        --tokenizer gpt2 \
        --input-path $INPUT_PATH/$dn/test.json \
        --output-path $OUTPUT_PATH/$dn/test
    uv run python ./tokenize_dataset_from_json.py \
        -sa \
        -mp \
        --tokenizer gpt2 \
        --input-path $INPUT_PATH/$dn/train.json \
        --output-path $OUTPUT_PATH/"$dn"_que/train
    uv run python ./tokenize_dataset_from_json.py \
        -sa \
        -mp \
        --tokenizer gpt2 \
        --input-path $INPUT_PATH/$dn/test.json \
        --output-path $OUTPUT_PATH/"$dn"_que/test
done
