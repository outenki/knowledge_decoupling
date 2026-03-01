#!/bin/bash

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH=$PROJECT_BASE_PATH/scripts/train/sft/gpt2
INPUT_MODEL=$PROJECT_BASE_PATH/output/gpt2/smolLM2/smolLM2_bs1024_dl0_ep1-ext_qa_test_ep3

/bin/bash "$SCRIPT_PATH"/sft_wo_context_train.sh \
   "$INPUT_MODEL" \
   "$INPUT_MODEL"-sft_qa_wo_context_train \
   3

/bin/bash "$SCRIPT_PATH"/sft_wo_context_test.sh \
   "$INPUT_MODEL" \
   "$INPUT_MODEL"-sft_qa_wo_context_test \
   3

/bin/bash "$SCRIPT_PATH"/sft_w_context_train.sh \
   "$INPUT_MODEL" \
   "$INPUT_MODEL"-sft_qa_w_context_train \
   3

/bin/bash "$SCRIPT_PATH"/sft_w_context_test.sh \
   "$INPUT_MODEL" \
   "$INPUT_MODEL"-sft_qa_w_context_test \
   3
