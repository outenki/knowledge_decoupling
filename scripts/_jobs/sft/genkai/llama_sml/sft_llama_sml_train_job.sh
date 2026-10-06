#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "gpu=4"
#PJM -e logs/sft_Llama3.2_1B_sml.log
#PJM -o logs/sft_Llama3.2_1B_sml.log
#PJM -N "sft_ls"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft.sh meta-llama/Llama-3.2-1B no_warmup/sml/SmolLM2-135M-20B-sml-bs4096