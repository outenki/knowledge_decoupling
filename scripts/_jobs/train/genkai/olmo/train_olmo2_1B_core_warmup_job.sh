#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=50:00:00"
#PJM -L "gpu=4"
#PJM -e logs/train_olmo2_1B_core_warmup.log
#PJM -o logs/train_olmo2_1B_core_warmup.log
#PJM -N "tr_oc_w"


module load cuda/12.8
source $HOME/.zshrc
cd $PROJECT_BASE_PATH/src/train

# export UV_OFFLINE=1
export WANDB_MODE=offline
uv run python train.py --config-name olmo2_1B-core-warmup base.path=$PROJECT_BASE_PATH
