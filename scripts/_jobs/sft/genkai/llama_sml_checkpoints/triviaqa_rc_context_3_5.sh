#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=50:00:00"
#PJM -L "gpu=4"
#PJM -e logs/triviaqa_rc_context_1_3.log
#PJM -o logs/triviaqa_rc_context_1_3.log
#PJM -N "sml_trc_3"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_triviaqa_rc_context.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_1
sh sft_triviaqa_rc_context.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_3
# sh sft_triviaqa_rc_context.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_5
# sh sft_triviaqa_rc_context.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_7
# sh sft_triviaqa_rc_context.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_9