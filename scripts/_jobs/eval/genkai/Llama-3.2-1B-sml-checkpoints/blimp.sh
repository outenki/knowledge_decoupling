#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=20:00:00"
#PJM -L "gpu=4"
#PJM -e logs/blimp.log
#PJM -o logs/blimp.log
#PJM -N "ev_lama_blimp"

source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096
sh lm_eval_blimp.sh $MODEL_PATH