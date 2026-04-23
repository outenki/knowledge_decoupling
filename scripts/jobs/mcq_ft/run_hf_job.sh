#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_hf.err
#PJM -o logs/run_hf.out
#PJM -N "run_hf"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/mcq_ft

sh run_hf.sh commonsense_qa
sh run_hf.sh arc_easy
sh run_hf.sh arc_challenge
sh run_hf.sh qasc
