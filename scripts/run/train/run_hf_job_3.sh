#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=16:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_hf_3.err
#PJM -o logs/run_hf_3.out
#PJM -N "run_hf_3"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/run/train

mkdir -p logs

sh run_hf.sh arc_challenge
