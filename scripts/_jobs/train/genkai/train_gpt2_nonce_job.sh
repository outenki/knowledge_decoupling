#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=70:00:00"
#PJM -L "gpu=4"
#PJM -e logs/train_gpt2_nonce.log
#PJM -o logs/train_gpt2_nonce.log
#PJM -N "tr_nonce"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/src/train

uv run python train.py --config-name gpt2-nonce base.path=$PROJECT_BASE_PATH
