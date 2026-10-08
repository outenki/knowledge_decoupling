#!/bin/bash
#PBS -q c30897g
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/clasheval.log
#PBS -N ms_cls


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_clasheval_mask.sh meta-llama/Llama-3.2-1B no_warmup/sml_mask/SmolLM2-135M-20B-sml_mask-bs4096