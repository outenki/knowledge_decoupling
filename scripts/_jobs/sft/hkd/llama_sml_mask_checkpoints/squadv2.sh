#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/squadv2.log
#PJM -o logs/squadv2.log
#PJM -N "ms_squadv2"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh squadv2.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/checkpoints/check_1
sh squadv2.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/checkpoints/check_3
sh squadv2.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/checkpoints/check_5
sh squadv2.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/checkpoints/check_7
sh squadv2.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/checkpoints/check_9