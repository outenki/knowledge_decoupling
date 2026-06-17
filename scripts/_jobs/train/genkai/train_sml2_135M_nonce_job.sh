#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=70:00:00"
#PJM -L "gpu=4"
#PJM -e logs/train_smolLM2_135M_nonce.log
#PJM -o logs/train_smolLM2_135M_nonce.log
#PJM -N "sml135-nonce"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/src/train

# export UV_OFFLINE=1
uv run python train.py --config-name SmolLM2-135M-nonce base.path=$PROJECT_BASE_PATH
