#!/bin/bash
#PBS -q lg
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/cnn_dailymail.log
#PBS -N "sft_cdm"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_cnn_dailymail.sh meta-llama/Llama-3.2-1B no_warmup/rnd/rnd