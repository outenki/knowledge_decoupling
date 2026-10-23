#! /bin/bash
uv run python $PROJECT_BASE_PATH/scripts/data_processing/split_based_on_key.py \
    --data-path $PROJECT_BASE_PATH/input/evaluate_data/jsonl/redirect_qa/data.jsonl \
    --train-ratio 0.9 \
    --key prompt \
    --output-dir $PROJECT_BASE_PATH/input/evaluate_data/jsonl/redirect_qa/

uv run python $PROJECT_BASE_PATH/scripts/data_processing/split_based_on_key.py \
    --data-path $PROJECT_BASE_PATH/input/evaluate_data/jsonl/popqa/data.jsonl \
    --train-ratio 0.9 \
    --key prompt \
    --output-dir $PROJECT_BASE_PATH/input/evaluate_data/jsonl/popqa/

# for sub in pod pob ins edu dob; do
#     echo ">>> Splitting google_re_${sub} into train and val"
#     uv run python $PROJECT_BASE_PATH/scripts/data_processing/split_based_on_context.py \
#         --data-path $PROJECT_BASE_PATH/input/evaluate_data/jsonl/google_re/${sub}/data.jsonl \
#         --train-ratio 0.9 \
#         --output-dir $PROJECT_BASE_PATH/input/evaluate_data/jsonl/google_re/${sub}
# done