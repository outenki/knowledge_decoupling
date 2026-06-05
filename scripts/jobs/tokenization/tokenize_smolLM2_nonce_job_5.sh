#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/tokenization_nonce_1024_5.out
#PJM -o logs/tokenization_nonce_1024_5.out
#PJM -N "tk_n_5"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/tokenization

sh ./tokenize_smolLM2_paralle_nonce.sh "Qwen/Qwen3.5-0.8B-Base" 1024 5
