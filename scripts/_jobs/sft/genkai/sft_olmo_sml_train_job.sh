#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "gpu=4"
#PJM -e logs/sft_OLMo-2-0425-1B_sml.log
#PJM -o logs/sft_OLMo-2-0425-1B_sml.log
#PJM -N "sft_olmo_sml"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft.sh allenai/OLMo-2-0425-1B SmolLM2-135M-20B-bs1024