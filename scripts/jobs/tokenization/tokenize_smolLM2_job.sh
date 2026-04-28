#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=36:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/tokenization.err
#PJM -o logs/tokenization.out
#PJM -N "tk"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/tokenization

sh ./tokenize_smolLM2.sh "Qwen/Qwen3.5-0.8B-Base"
