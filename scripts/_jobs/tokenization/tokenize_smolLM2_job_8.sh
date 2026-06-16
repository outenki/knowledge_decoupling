#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/tokenization_1024_8.out
#PJM -o logs/tokenization_1024_8.out
#PJM -N "tk_s_8"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/tokenization

sh ./tokenize_smolLM2_paralle.sh 1024 8
