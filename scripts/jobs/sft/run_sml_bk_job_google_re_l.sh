#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_sml_bk_job_google_re_long.err
#PJM -o logs/run_sml_bk_job_google_re_long.out
#PJM -N "sml_bk_grel"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/sft

sh run_sml_bk.sh google_re_long concat
sh run_sml_bk.sh google_re_long chat_template