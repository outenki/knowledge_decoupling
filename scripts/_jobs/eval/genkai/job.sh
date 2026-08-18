#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=10:00:00"
#PJM -L "gpu=4"
#PJM -e logs/context_gain.log
#PJM -o logs/context_gain.log
#PJM -N "eval_cg"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

sh lm_eval_test.sh