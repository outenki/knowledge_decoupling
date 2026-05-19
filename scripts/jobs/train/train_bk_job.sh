#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=50:00:00"
#PJM -L "gpu=4"
#PJM -e logs/train_qwen3.5-0.8B-Base_bk.err
#PJM -o logs/train_qwen3.5-0.8B-Base_bk.out
#PJM -N "tr_qw_bk"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/train

sh ./train_bk.sh
