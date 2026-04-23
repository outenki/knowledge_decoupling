#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_eval_qa_all.err
#PJM -o logs/run_eval_qa_all.out
#PJM -N "run_eval_qa_all"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/run/eval

sh run_eval_qa_all.sh

