#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=50:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/core_7.out
#PJM -o logs/core_7.out
#PJM -N "core_7"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/core_data

sh ./core_smolLM2_paralle_nonce.sh 7
