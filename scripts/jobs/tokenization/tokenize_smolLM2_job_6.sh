#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=36:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/tokenization_4096_6.err
#PJM -o logs/tokenization_4096_6.out
#PJM -N "tk_6"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/tokenization

sh ./tokenize_smolLM2_paralle.sh "Qwen/Qwen3.5-0.8B-Base" 6
