#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_sml_bk_job_qasc.err
#PJM -o logs/run_sml_bk_job_qasc.out
#PJM -N "sml_bk_qasc"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/sft

sh run_sml_bk.sh qasc concat
sh run_sml_bk.sh qasc chat_template