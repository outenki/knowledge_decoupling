#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/olme2-sml-job_4.out
#PJM -o logs/olme2-sml-job_4.out
#PJM -N "ol_sml_04"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/tokenization/tokenize_limit

sh ./tokenize_smolLM2_paralle.sh allenai/OLMo-2-0425-1B 4
