#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=30:00:00"
#PJM -L "gpu=4"
#PJM -e logs/train_Llama3.2_1B_sml_mask.log
#PJM -o logs/train_Llama3.2_1B_sml_mask.log
#PJM -N "tr_lsm"


module load cuda/12.8
source $HOME/.zshrc
cd $PROJECT_BASE_PATH/src/train

# export UV_OFFLINE=1
export WANDB_MODE=offline
uv run python train.py --config-name Llama3.2_1B-sml-mask base.path=$PROJECT_BASE_PATH
