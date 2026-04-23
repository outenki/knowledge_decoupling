#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/run_hf_job_arc_challenge.err
#PJM -o logs/run_hf_job_arc_challenge.out
#PJM -N "eval_hf_ac"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/jobs/sft

sh evaluate_hf.sh arc_challenge concat
sh evaluate_hf.sh arc_challenge chat_template
