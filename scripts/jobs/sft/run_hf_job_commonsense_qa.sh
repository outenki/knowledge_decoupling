#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_hf_job_commonsense_qa.err
#PJM -o logs/run_hf_job_commonsense_qa.out
#PJM -N "sft_hf_cqa"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/sft

sh run_hf.sh commonsense_qa concat
