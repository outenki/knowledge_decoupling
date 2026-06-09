#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=100:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o log/run_hf.log


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/src/train

uv run python train.py --config-name qwen_sml_20B
