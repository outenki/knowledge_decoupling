#!/bin/bash
#PBS -q c30897g
#PBS -l select=1:ngpus=4
#PBS -l walltime=50:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/piqa.log
#PBS -N "rid_piqa"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_piqa.sh meta-llama/Llama-3.2-1B seed_41/no_warmup/rnd_id/SmolLM2-135M-20B-rnd_id-bs4096
sh sft_piqa.sh meta-llama/Llama-3.2-1B seed_43/no_warmup/rnd_id/SmolLM2-135M-20B-rnd_id-bs4096