#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=16:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_hf_1.err
#PJM -o logs/run_hf_1.out
#PJM -N "run_hf_1"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/run/train

mkdir -p logs

sh run_hf_ep1-base.sh commonsense_qa
