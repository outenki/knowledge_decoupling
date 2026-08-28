#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/olmo2-sml-mask-job_1.out
#PJM -o logs/olmo2-sml-mask-job_1.out
#PJM -N "ol_msk_01"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/tokenization/tokenize_limit

sh ./tokenize_smolLM2_mask_paralle.sh allenai/OLMo-2-0425-1B 1
