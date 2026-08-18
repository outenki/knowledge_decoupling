#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=10:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/qwen2.5_1.out
#PJM -o logs/qwen2.5_1.out
#PJM -N "tk_qw_1"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/tokenization/tokenize_parts

sh ./tokenize.sh Qwen/Qwen2.5-0.5B 1
