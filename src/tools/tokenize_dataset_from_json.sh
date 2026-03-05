#! /bin/bash
# For extensive pretraining
INPUT_PATH=$HOME/projects/knowledge_decoupling/input/evaluate_data/unformated/

OUTPUT_PATH=$HOME/projects/knowledge_decoupling/input/tokenized/gpt2/ext
for dn in \
    google_re_long_context \
    google_re_short_context \
    commonsense_qa
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
        --output-path $OUTPUT_PATH/$dn-ans/train
    uv run python ./tokenize_dataset_from_json.py \
        -sa \
        --tokenizer gpt2 \
        --input-path $INPUT_PATH/$dn/test.json \
        --output-path $OUTPUT_PATH/$dn-ans/test
done

OUTPUT_PATH=$HOME/projects/knowledge_decoupling/input/tokenized/gpt2/sft
for dn in \
    google_re_long_context \
    google_re_short_context \
    commonsense_qa
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
        --output-path $OUTPUT_PATH/$dn-ans/train
    uv run python ./tokenize_dataset_from_json.py \
        -sa \
        -mp \
        --tokenizer gpt2 \
        --input-path $INPUT_PATH/$dn/test.json \
        --output-path $OUTPUT_PATH/$dn-ans/test
done
