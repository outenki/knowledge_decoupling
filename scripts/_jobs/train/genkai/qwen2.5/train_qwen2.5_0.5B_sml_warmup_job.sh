#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=50:00:00"
#PJM -L "gpu=4"
#PJM -e logs/train_qwen2.5_0.5B_sml_warmup.log
#PJM -o logs/train_qwen2.5_0.5B_sml_warmup.log
#PJM -N "tr_qs_w"

module load cuda/12.8
source $HOME/.zshrc
cd $PROJECT_BASE_PATH/src/train

# export UV_OFFLINE=1
export WANDB_MODE=offline
uv run python train.py --config-name qwen2.5_0.5B-sml-warmup base.path=$PROJECT_BASE_PATH
