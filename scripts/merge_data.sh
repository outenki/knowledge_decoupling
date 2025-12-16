#!/bin/bash

BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
INPUT_PATH=$BASE_PATH/data/SmolLM2/sents/mn_3/tokenized-nonce/HuggingFaceTB/SmolLM2-135M/tokenized_bs1024
OUTPUT_PATH=$BASE_PATH/data/SmolLM2/sents/mn_3/tokenized-nonce/smolLm2-tokenized-bs1024

/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/merge_dataset.py \
    -pr 0 100 \
    -dd $INPUT_PATH \
    -o $OUTPUT_PATH
