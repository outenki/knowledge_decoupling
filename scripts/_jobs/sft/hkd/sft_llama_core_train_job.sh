#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=50:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/llama_sft_core.log
#PBS -N sft_llama_core


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

# sh sft_core.sh meta-llama/Llama-3.2-1B SmolLM2-135M-20B-core_ent-bs4096
sh sft.sh meta-llama/Llama-3.2-1B SmolLM2-135M-20B-core_ent-bs4096
