#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/squadv2_1.log
#PJM -o logs/squadv2_1.log
#PJM -N "sml_sq_1_3"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_squadv2.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_1
sh sft_squadv2.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_3
# sh sft_squadv2.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_5
# sh sft_squadv2.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_7
# sh sft_squadv2.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_9