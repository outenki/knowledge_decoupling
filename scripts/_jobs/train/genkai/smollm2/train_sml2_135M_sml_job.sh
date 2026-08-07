#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=70:00:00"
#PJM -L "gpu=4"
#PJM -e logs/train_smolLM2_135M_sml.log
#PJM -o logs/train_smolLM2_135M_sml.log
#PJM -N "sml135"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/src/train

uv run python train.py --config-name SmolLM2-135M-sml base.path=$PROJECT_BASE_PATH
