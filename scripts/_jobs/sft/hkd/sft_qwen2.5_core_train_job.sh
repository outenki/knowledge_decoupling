#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=50:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/qwen_sft_core.log


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft/hkd

sh sft_core.sh Qwen/Qwen2.5-0.5B