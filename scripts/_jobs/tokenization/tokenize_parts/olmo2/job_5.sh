#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=10:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/olmo2_1024_5.out
#PJM -o logs/olmo2_1024_5.out
#PJM -N "tk_ol_5"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/tokenization/tokenize_parts

sh ./tokenize.sh allenai/OLMo-2-0425-1B 5
