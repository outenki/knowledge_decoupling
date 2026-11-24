#!/bin/bash

config_name="${1:?Usage: run_llama3_2_1b.sh CONFIG_NAME SEED}"
seed="${2:?Usage: run_llama3_2_1b.sh CONFIG_NAME SEED}"

module load cuda/12.8
source "$HOME/.zshrc"
cd "$PROJECT_BASE_PATH/src/train"

export WANDB_MODE=offline
uv run python train.py \
  --config-name "$config_name" \
  "base.path=$PROJECT_BASE_PATH" \
  "seed=$seed"
