#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_hf_1.out
#PJM -o logs/run_hf_1.err
#PJM -N "run_hf_1"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/run/train

mkdir -p logs

sh run_hf.sh commonsense_qa
