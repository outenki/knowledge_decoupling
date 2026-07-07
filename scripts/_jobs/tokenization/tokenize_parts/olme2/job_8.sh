#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=10:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/olme2_1024_8.out
#PJM -o logs/olme2_1024_8.out
#PJM -N "tk_ol_8"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/tokenization/tokenize_parts

sh ./tokenize.sh allenai/OLMo-2-0425-1B 8
