#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=100:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/train_smolLM2_135M_sml.log


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/src/train

# export UV_OFFLINE=1
export WANDB_MODE=offline
uv run python train.py --config-name SmolLM2-135M-sml base.path=$PROJECT_BASE_PATH
