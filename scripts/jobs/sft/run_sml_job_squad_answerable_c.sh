#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=36:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_sml_job_squad_v2_answerable_concat.err
#PJM -o logs/run_sml_job_squad_v2_answerable_concat.out
#PJM -N "sml_sq_c"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/sft

sh run_sml.sh squad_v2_answerable concat
# sh run_sml.sh squad_v2_answerable chat_template