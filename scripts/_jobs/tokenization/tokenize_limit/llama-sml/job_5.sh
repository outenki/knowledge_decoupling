#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/llama-sml-job_5.out
#PJM -o logs/llama-sml-job_5.out
#PJM -N "llama-sml_05"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/tokenization/tokenize_limit

sh ./tokenize_smolLM2_paralle.sh meta-llama/Llama-3.2-1B 5
