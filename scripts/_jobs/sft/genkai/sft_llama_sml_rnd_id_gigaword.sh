#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/sft_Llama3.2_1B_sml_rnd_id_gigaword.log
#PJM -o logs/sft_Llama3.2_1B_sml_rnd_id_gigaword.log
#PJM -N "sft_gw"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_gigaword.sh meta-llama/Llama-3.2-1B no_warmup/sml_rnd_id/SmolLM2-135M-20B-rnd_id-bs4096