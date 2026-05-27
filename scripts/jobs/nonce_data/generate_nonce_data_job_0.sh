#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/nonce_bank_2000000_0.out
#PJM -o logs/nonce_bank_2000000_0.out
#PJM -N "nd_0"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/nonce_data

sh ./generate_nonce_data_paralle.sh 0
