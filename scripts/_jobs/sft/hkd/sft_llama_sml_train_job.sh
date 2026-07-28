#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=50:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/llama_sft_sml.log
#PBS -N sft_llama_sml


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft/hkd

sh sft.sh meta-llama/Llama-3.2-1B SmolLM2-135M-20B-bs1024
# sh sft_core.sh meta-llama/Llama-3.2-1B SmolLM2-135M-20B-bs1024