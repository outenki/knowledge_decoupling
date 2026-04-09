#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=10:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o log/sft_gpt2_rnd.log.$PBS_JOBID


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/train/sft/gpt2

mkdir -p log

sh sft_gpt2_rnd.sh
