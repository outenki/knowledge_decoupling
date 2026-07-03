#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=10:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/gpt2_1024_8.out
#PJM -o logs/gpt2_1024_8.out
#PJM -N "tk_gpt2_8"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/tokenization/tokenize

sh ./tokenize.sh openai-community/gpt2 8
