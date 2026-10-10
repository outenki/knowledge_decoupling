#!/bin/bash
#PBS -q c30897g
#PBS -l select=1:ngpus=4
#PBS -l walltime=50:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/clasheval_1_3.log
#PBS -N "sml_cls_1_3"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_clasheval.sh meta-llama/Llama-3.2-1B no_warmup/sml/SmolLM2-135M-20B-sml-bs4096
sh sft_clasheval.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_1
sh sft_clasheval.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_3