#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/nonce_data_2000000_4.out
#PJM -o logs/nonce_data_2000000_4.out
#PJM -N "nd_4"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/nonce_data

sh ./generate_nonce_data_paralle.sh 4
