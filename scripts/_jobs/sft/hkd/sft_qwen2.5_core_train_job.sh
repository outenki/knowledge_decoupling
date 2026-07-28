#!/bin/bash
#PBS -q lg
#PBS -l select=1:ngpus=4
#PBS -l walltime=50:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/qwen_sft_core.log
#PBS -N sft_qwen2.5_core


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft/hkd

sh sft.sh Qwen/Qwen2.5-0.5B SmolLM2-135M-20B-core-bs1024
# sh sft_core.sh Qwen/Qwen2.5-0.5B SmolLM2-135M-20B-core-bs1024