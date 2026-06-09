#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=70:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/nonce_data_6.out
#PJM -o logs/nonce_data_6.out
#PJM -N "nd_6"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/nonce_data

sh ./generate_nonce_data_paralle.sh 6
