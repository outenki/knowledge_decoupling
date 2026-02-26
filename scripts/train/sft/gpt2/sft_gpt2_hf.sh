#!/bin/bash

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH=$PROJECT_BASE_PATH/scripts/train/sft/gpt2

# /bin/bash $SCRIPT_PATH/sft_w_context_test.sh \
#     gpt2 \
#     $PROJECT_BASE_PATH/output/gpt2/HuggingFace/hf-sft_qa_context_test \
#     3

# /bin/bash $SCRIPT_PATH/sft_w_context_train.sh \
#     gpt2 \
#     $PROJECT_BASE_PATH/output/gpt2/HuggingFace/hf-sft_qa_context_train \
#     3

# /bin/bash $SCRIPT_PATH/sft_wo_context_test.sh \
#     gpt2 \
#     $PROJECT_BASE_PATH/output/gpt2/HuggingFace/hf-sft_qa_wo_context_test \
#     3

/bin/bash "$SCRIPT_PATH"/sft_wo_context_train.sh \
    gpt2 \
    "$PROJECT_BASE_PATH"/output/gpt2/HuggingFace/hf-sft_qa_wo_context_train \
    3 \
    "$PROJECT_BASE_PATH"/output/gpt2/HuggingFace/hf-sft_qa_wo_context_train/checkpoint-3892 \
