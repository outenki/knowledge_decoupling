#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -e logs/sft_Qwen2.5-0.5B_core.log
#PJM -o logs/sft_Qwen2.5-0.5B_core.log
#PJM -N "sft_qwen2.5_core"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft.sh Qwen/Qwen2.5-0.5B SmolLM2-135M-20B-core_ent-bs4096