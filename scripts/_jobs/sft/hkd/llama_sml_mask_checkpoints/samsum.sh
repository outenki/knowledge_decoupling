#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/samsum.log
#PJM -o logs/samsum.log
#PJM -N "ms_samsum"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_samsum.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/checkpoints/check_1
sh sft_samsum.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/checkpoints/check_3
sh sft_samsum.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/checkpoints/check_5
sh sft_samsum.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/checkpoints/check_7
sh sft_samsum.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/checkpoints/check_9