#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=50:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/olmo_sft_core.log
#PBS -N sft_olmo_core


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

# sh sft_core.sh allenai/OLMo-2-0425-1B SmolLM2-135M-20B-core_ent-bs4096
sh sft.sh allenai/OLMo-2-0425-1B warmup/core_ent/SmolLM2-135M-20B-core_ent-bs4096
