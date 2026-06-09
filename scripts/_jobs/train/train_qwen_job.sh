#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "gpu=4"
#PJM -e logs/train_qwen3.5-0.8B-Base-sml_20B.out
#PJM -o logs/train_qwen3.5-0.8B-Base-sml_20B.out
#PJM -N "tr_qw"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/src/train

uv run python train.py --config-name qwen_sml_20B
