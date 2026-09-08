#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/generate_squadv2.log
#PJM -o logs/generate_squadv2.log
#PJM -N "gen_squadv2"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/data_processing

OUTPUT_PATH=$PROJECT_BASE_PATH/input/evaluate_data
EXT_TRAINING_PATH=$PROJECT_BASE_PATH/data/ext
SFT_TRAINING_PATH=$PROJECT_BASE_PATH/data/sft

echo ">>> squadv2_core"
uv run python generate_qa_data.py \
    -dn squadv2 \
    -o $OUTPUT_PATH/jsonl/squadv2_ent_id \
    --core-replace \
    --aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
    -at 10 \
    --ent-generator "ENT_ID" \
    --unk-generator "UNK_ID" \
    --core-count \
    --core-delimiter "<>" \
    -ot jsonl