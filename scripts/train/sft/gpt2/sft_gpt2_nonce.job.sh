#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=10:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o log/sft_gpt2_nonce.log


source $HOME/.zshrc
cd /lustre1/work/c30897/wtq/projects/knowledge_decoupling/scripts/train/sft/gpt2

mkdir -p log

sh sft_gpt2_nonce.sh
