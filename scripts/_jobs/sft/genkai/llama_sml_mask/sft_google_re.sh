#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/google_re.log
#PJM -o logs/google_re.log
#PJM -N "ms_gre"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_google_re_mask.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096