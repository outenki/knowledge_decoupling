# !/bin/bash
TOKENIZER="Qwen/Qwen3.5-0.8B-Base"
for data_name in \
    commonsense_qa \
    qasc \
    arc_easy \
    arc_challenge
do
    echo ">>> Tokenizing dataset: $data_name"
    uv run python tokenize_dataset_from_json_mcq.py \
        -tk "$TOKENIZER" \
        -i ../../input/evaluate_data/unformated/$data_name/train.json \
        -o ../../input/tokenized/Qwen/Qwen3.5-0.8B/mcq_ft/$data_name/train
    uv run python tokenize_dataset_from_json_mcq.py \
        -tk "$TOKENIZER" \
        -i ../../input/evaluate_data/unformated/$data_name/test.json \
        -o ../../input/tokenized/Qwen/Qwen3.5-0.8B/mcq_ft/$data_name/test
    echo
done