#!/bin/bash
#PBS -q lg
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/sft_Llama3.2_1B_sml_rnd_id.log
#PBS -N sft_lsmr

source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft.sh meta-llama/Llama-3.2-1B no_warmup/sml_rnd_id/SmolLM2-135M-20B-rnd_id-bs4096