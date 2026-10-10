#!/bin/bash
#PBS -q c30897g
#PBS -l select=1:ngpus=4
#PBS -l walltime=50:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/google_re_5_9.log
#PBS -N "rid_gre_5_9"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/_jobs/sft

sh sft_google_re.sh meta-llama/Llama-3.2-1B no_warmup/sml_rnd_id/checkpoints/check_5
sh sft_google_re.sh meta-llama/Llama-3.2-1B no_warmup/sml_rnd_id/checkpoints/check_7
sh sft_google_re.sh meta-llama/Llama-3.2-1B no_warmup/sml_rnd_id/checkpoints/check_9