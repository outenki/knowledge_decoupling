#!/bin/bash
#PBS -q c30897g
#PBS -l select=1:ngpus=4
#PBS -l walltime=160:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o "logs/train_llama3.2_1B_sml_41.log"
#PBS -N "tr_41_ls"

module load cuda/12.8
source $HOME/.zshrc
cd $PROJECT_BASE_PATH/src/train

# export UV_OFFLINE=1
export WANDB_MODE=offline
uv run python train.py --config-name Llama3.2_1B-sml base.path=$PROJECT_BASE_PATH seed=41