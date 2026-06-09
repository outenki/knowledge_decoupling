#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=72:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/filter_9.err
#PJM -o logs/filter_9.out
#PJM -N "flt_9"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/filter_dataset

sh ./filter_dataset.sh 9
