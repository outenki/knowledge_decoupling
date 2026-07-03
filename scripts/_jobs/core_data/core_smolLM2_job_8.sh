#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=50:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/core_8.out
#PJM -o logs/core_8.out
#PJM -N "core_8"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/core_data

sh ./core_smolLM2_paralle_nonce.sh 8
