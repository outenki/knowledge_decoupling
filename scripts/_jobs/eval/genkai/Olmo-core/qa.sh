#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=70:00:00"
#PJM -L "gpu=4"
#PJM -e logs/qa.log
#PJM -o logs/qa.log
#PJM -N "eval_olmo_qa"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/allenai/OLMo-2-0425-1B/SmolLM2-135M-20B-core_ent-bs4096
sh lm_eval_qa.sh $MODEL_PATH