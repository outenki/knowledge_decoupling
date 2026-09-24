#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/sft_Llama3.2_1B_sml_mask_triva_context.log
#PJM -o logs/sft_Llama3.2_1B_sml_mask_triva_context.log
#PJM -N "sft_trc"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_triviaqa_rc_context.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096