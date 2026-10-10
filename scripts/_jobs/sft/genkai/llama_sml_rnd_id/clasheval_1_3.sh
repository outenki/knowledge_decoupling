#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/clasheval_1_3.log
#PJM -o logs/clasheval_1_3.log
#PJM -N "rid_cls_1_3"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_clasheval.sh meta-llama/Llama-3.2-1B no_warmup/sml_rnd_id/SmolLM2-135M-20B-rnd_id-bs4096
sh sft_clasheval.sh meta-llama/Llama-3.2-1B no_warmup/sml_rnd_id/checkpoints/check_1
sh sft_clasheval.sh meta-llama/Llama-3.2-1B no_warmup/sml_rnd_id/checkpoints/check_3