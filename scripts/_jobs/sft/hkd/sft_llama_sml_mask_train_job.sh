#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=50:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/sft_Llama3.2_1B_sml_mask.log
#PBS -N sft_lsm

source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096