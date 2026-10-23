#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -j
#PJM -o logs/redirect_qa.log
#PJM -N "sft_rdqa"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_redirect_qa.sh meta-llama/Llama-3.2-1B no_warmup/sml_rnd_id/SmolLM2-135M-20B-rnd_id-bs4096