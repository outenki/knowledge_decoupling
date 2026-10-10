#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/nq_swap_1_3.log
#PJM -o logs/nq_swap_1_3.log
#PJM -N "sml_nq_1_3"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_nq_swap.sh meta-llama/Llama-3.2-1B no_warmup/sml/SmolLM2-135M-20B-sml-bs4096
sh sft_nq_swap.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_1
sh sft_nq_swap.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_3