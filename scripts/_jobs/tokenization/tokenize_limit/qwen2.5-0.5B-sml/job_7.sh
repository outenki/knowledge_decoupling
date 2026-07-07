#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/qwen2.5-0.5B-sml-job_7.out
#PJM -o logs/qwen2.5-0.5B-sml-job_7.out
#PJM -N "qw_sml_7"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/tokenization/tokenize_limit

sh ./tokenize_smolLM2_paralle.sh Qwen/Qwen3.5-0.8B-Base 7
