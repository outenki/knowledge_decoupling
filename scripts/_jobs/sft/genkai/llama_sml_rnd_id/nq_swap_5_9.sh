#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/nq_swap_5_9.log
#PJM -o logs/nq_swap_5_9.log
#PJM -N "rid_nq_5_9"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_nq_swap.sh meta-llama/Llama-3.2-1B no_warmup/sml_rnd_id/checkpoints/check_5
sh sft_nq_swap.sh meta-llama/Llama-3.2-1B no_warmup/sml_rnd_id/checkpoints/check_7
sh sft_nq_swap.sh meta-llama/Llama-3.2-1B no_warmup/sml_rnd_id/checkpoints/check_9