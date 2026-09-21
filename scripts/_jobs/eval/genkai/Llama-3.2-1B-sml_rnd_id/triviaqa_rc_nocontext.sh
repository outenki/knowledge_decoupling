#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=50:00:00"
#PJM -L "gpu=4"
#PJM -e logs/triviaqa_rc_nocontext.log
#PJM -o logs/triviaqa_rc_nocontext.log
#PJM -N "ev_lm_trnc"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml_rnd_id/SmolLM2-135M-20B-rnd_id-bs4096
sh lm_eval_triviaqa_rc_nocontext.sh $MODEL_PATH