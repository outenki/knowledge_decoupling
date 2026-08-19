#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=60:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o "logs/train_olmo2_1B_core_warmup.log"
#PBS -N "tr_oc_w"

module load cuda/12.8
source $HOME/.zshrc
cd $PROJECT_BASE_PATH/src/train

# export UV_OFFLINE=1
export WANDB_MODE=offline
uv run python train.py --config-name olmo2_1B-core-warmup base.path=$PROJECT_BASE_PATH