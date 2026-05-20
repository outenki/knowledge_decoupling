#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_sml_job_google_re_short.err
#PJM -o logs/run_sml_job_google_re_short.out
#PJM -N "sml_gres"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/sft

sh run_sml.sh google_re_short concat
sh run_sml.sh google_re_short chat_template