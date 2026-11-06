#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=50:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/google_boolq.log
#PBS -N google_boolq



source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=
sh lm_eval_google_boolq.sh $PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml/checkpoints/check_1
sh lm_eval_google_boolq.sh $PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml/checkpoints/check_3
sh lm_eval_google_boolq.sh $PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml/checkpoints/check_5
sh lm_eval_google_boolq.sh $PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml/checkpoints/check_7
sh lm_eval_google_boolq.sh $PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml/checkpoints/check_9