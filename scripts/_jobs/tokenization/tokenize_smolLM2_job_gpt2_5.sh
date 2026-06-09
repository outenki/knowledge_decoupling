#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=50:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/tokenization_1024_5.out
#PJM -o logs/tokenization_1024_5.out
#PJM -N "tkg_5"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/tokenization

sh ./tokenize_smolLM2_paralle.sh "gpt2" 1024 5
