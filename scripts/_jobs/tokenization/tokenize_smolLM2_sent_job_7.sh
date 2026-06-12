#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/tokenization_sent_1024_7.out
#PJM -o logs/tokenization_sent_1024_7.out
#PJM -N "tk_s_7"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/tokenization

sh ./tokenize_smolLM2_paralle_sent.sh "HuggingFaceTB/SmolLM2-135M" 1024 7
