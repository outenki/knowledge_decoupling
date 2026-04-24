#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_hf_gpt2_job_commonsense_qa.err
#PJM -o logs/run_hf_gpt2_job_commonsense_qa.out
#PJM -N "gpt2_cq"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/sft

sh run_hf_gpt2.sh commonsense_qa concat