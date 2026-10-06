#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/triviaqa_rc_context_1.log
#PJM -o logs/triviaqa_rc_context_1.log
#PJM -N "ms_trc_1"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_triviaqa_rc_context.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/checkpoints/check_1
sh sft_triviaqa_rc_context.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/checkpoints/check_3