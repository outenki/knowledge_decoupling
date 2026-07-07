#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=10:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/sml_1024_6.out
#PJM -o logs/sml_1024_6.out
#PJM -N "tk_sml_6"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/tokenization/tokenize

sh ./tokenize.sh HuggingFaceTB/SmolLM2-135M 6
