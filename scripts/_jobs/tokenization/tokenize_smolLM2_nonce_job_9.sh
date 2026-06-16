#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/tokenization_nonce_1024_9.out
#PJM -o logs/tokenization_nonce_1024_9.out
#PJM -N "tk_n_9"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/tokenization

sh ./tokenize_smolLM2_paralle_nonce.sh 1024 9
