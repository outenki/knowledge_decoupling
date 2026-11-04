#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=50:00:00"
#PJM -L "gpu=4"
#PJM -e logs/qa.log
#PJM -o logs/qa.log
#PJM -N "evsml_qa"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml
sh lm_eval_qa.sh $MODEL_PATH/SmolLM2-135M-20B-sml-bs4096
sh lm_eval_qa.sh $MODEL_PATH/checkpoints/check_1
sh lm_eval_qa.sh $MODEL_PATH/checkpoints/check_3
sh lm_eval_qa.sh $MODEL_PATH/checkpoints/check_5
sh lm_eval_qa.sh $MODEL_PATH/checkpoints/check_7
sh lm_eval_qa.sh $MODEL_PATH/checkpoints/check_9