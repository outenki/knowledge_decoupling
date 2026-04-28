#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_hf_gpt2_job_arc_easy.err
#PJM -o logs/run_hf_gpt2_job_arc_easy.out
#PJM -N "gpt2_ae"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/sft

sh run_hf_gpt2.sh arc_easy concat