#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=36:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/nonce_bank_2000000_9.err
#PJM -o logs/nonce_bank_2000000_9.out
#PJM -N "nb_9"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/nonce_data

sh ./generate_nonce_bank_paralle.sh 9
