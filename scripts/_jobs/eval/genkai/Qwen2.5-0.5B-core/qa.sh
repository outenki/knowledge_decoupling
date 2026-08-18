#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=70:00:00"
#PJM -L "gpu=4"
#PJM -e logs/qa.log
#PJM -o logs/qa.log
#PJM -N "eval_qwen_qa"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/Qwen/Qwen2.5-0.5B/SmolLM2-135M-20B-core_ent-bs4096
sh lm_eval_qa.sh $MODEL_PATH