#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/clasheval.log
#PJM -o logs/clasheval.log
#PJM -N "ri_clm"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_clasheval_mask.sh meta-llama/Llama-3.2-1B no_warmup/sml_rnd_id/SmolLM2-135M-20B-rnd_id-bs4096