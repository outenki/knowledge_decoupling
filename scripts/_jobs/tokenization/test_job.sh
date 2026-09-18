#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/script.out
#PJM -o logs/script.out
#PJM -N "script"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/data_processing

sh ./tokenize_dataset_from_json.sh
