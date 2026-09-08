#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=60:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/generate_trivia_rc_ctx.log
#PJM -o logs/generate_trivia_rc_ctx.log
#PJM -N "gen_trivia_rc_ctx"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/data_processing

OUTPUT_PATH=$PROJECT_BASE_PATH/input/evaluate_data
EXT_TRAINING_PATH=$PROJECT_BASE_PATH/data/ext
SFT_TRAINING_PATH=$PROJECT_BASE_PATH/data/sft

echo ">>> triviaqa_rc_core"
uv run python generate_qa_data.py \
    -dn triviaqa_rc_context \
    -o $OUTPUT_PATH/jsonl/triviaqa_rc_ent_id \
    --core-replace \
    --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
    -at 10 \
    --ent-generator "ENT_ID" \
    --unk-generator "UNK_ID" \
    --core-count \
    --core-delimiter "<>" \
    -ot jsonl
