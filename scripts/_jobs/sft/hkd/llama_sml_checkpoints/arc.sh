#!/bin/bash
#PBS -q c30897g
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/arc.log
#PBS -N smlc_arc


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_arc_easy_challenge.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_1
sh sft_arc_easy_challenge.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_3
sh sft_arc_easy_challenge.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_5
sh sft_arc_easy_challenge.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_7
sh sft_arc_easy_challenge.sh meta-llama/Llama-3.2-1B no_warmup/sml/checkpoints/check_9