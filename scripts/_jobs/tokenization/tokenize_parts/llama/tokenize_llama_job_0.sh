#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=10:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/llama_0.out
#PJM -o logs/llama_0.out
#PJM -N "tk_llama_0"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/tokenization/tokenize

sh ./tokenize.sh meta-llama/Llama-3.2-1B 0
