#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_sml_job_arc_easy.err
#PJM -o logs/run_sml_job_arc_easy.out
#PJM -N "sml_ae"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/sft

sh run_sml.sh arc_easy concat
sh run_sml.sh arc_easy chat_template