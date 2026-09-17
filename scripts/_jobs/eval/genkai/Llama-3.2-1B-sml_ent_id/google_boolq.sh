#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=50:00:00"
#PJM -L "gpu=4"
#PJM -e logs/google_boolq.log
#PJM -o logs/google_boolq.log
#PJM -N "ev_lei_gb"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml_ent_id/SmolLM2-135M-20B-core_ent_id-bs4096
sh lm_eval_google_boolq.sh $MODEL_PATH