#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_sml_job_commonsense_qa.err
#PJM -o logs/run_sml_job_commonsense_qa.out
#PJM -N "sml_cq"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/sft

sh run_sml.sh commonsense_qa concat
sh run_sml.sh commonsense_qa chat_template