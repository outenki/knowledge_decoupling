#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=50:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/gpt2_sft_hf.log
#PBS -N sft_gpt2_hf


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft/hkd

sh sft.sh openai-community/gpt2 hf_full
sh sft_core.sh openai-community/gpt2 hf_full