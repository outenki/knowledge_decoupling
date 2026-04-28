#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=72:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/filter_14.err
#PJM -o logs/filter_14.out
#PJM -N "flt_14"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/filter_dataset

sh ./filter_dataset.sh 14
