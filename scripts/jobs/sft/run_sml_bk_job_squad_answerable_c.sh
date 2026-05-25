#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=48:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_sml_bk_job_squad_v2_answerable_concat.err
#PJM -o logs/run_sml_bk_job_squad_v2_answerable_concat.out
#PJM -N "sml_bk_sqc"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/sft

sh run_sml_bk.sh squad_v2_answerable concat
# sh run_sml_bk.sh squad_v2_answerable chat_template