#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "gpu=4"
#PJM -e logs/train_qwen3.5-0.8B-Base.err
#PJM -o logs/train_qwen3.5-0.8B-Base.out
#PJM -N "tr_qw"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/train

sh ./train.sh
