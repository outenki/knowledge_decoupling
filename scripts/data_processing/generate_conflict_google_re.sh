#!/bin/bash

# for sub in pod pob ins dob; do
#     echo ">>> Generating conflicts for google_re_${sub}"
#     uv run python $PROJECT_BASE_PATH/scripts/data_processing/generate_conflict_google_re.py \
#         -i $PROJECT_BASE_PATH/input/evaluate_data/jsonl/google_re/${sub}/val.jsonl \
#         -o $PROJECT_BASE_PATH/input/evaluate_data/jsonl/google_re/${sub}/val_conflict.jsonl
# done
uv run python $PROJECT_BASE_PATH/scripts/data_processing/generate_conflict_google_re.py \
    -i $PROJECT_BASE_PATH/input/evaluate_data/jsonl/google_re/val.jsonl \
    -o $PROJECT_BASE_PATH/input/evaluate_data/jsonl/google_re/val_conflict.jsonl