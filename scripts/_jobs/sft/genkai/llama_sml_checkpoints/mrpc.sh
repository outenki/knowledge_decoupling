#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=50:00:00"
#PJM -L "gpu=4"
#PJM -j
#PJM -o logs/mrpc.log
#PJM -N "smlc_mrpc"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

# sh sft_mrpc.sh meta-llama/Llama-3.2-1B no_warmup/sml/SmolLM2-135M-20B-sml-bs4096
sh sft_mrpc.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_1
sh sft_mrpc.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_3
sh sft_mrpc.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_5
sh sft_mrpc.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_7
sh sft_mrpc.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_9