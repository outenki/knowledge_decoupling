#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/winogrande.log
#PJM -o logs/winogrande.log
#PJM -N "sft_wino"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_winogrande.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_1
sh sft_winogrande.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_3
sh sft_winogrande.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_5
sh sft_winogrande.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_7
sh sft_winogrande.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_9