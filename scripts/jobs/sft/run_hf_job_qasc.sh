#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_hf_job_qasc.err
#PJM -o logs/run_hf_job_qasc.out
#PJM -N "sft_hf_qasc"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/sft

sh run_hf.sh qasc chat_template
