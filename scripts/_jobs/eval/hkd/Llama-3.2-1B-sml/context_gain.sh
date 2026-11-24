#!/bin/bash
#PBS -q c30897g
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/context_gain.log
#PBS -N cg_llama_sml


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

sh lm_eval_context_gain.sh $PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/seed_41/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096
sh lm_eval_context_gain.sh $PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/seed_42/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096
sh lm_eval_context_gain.sh $PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/seed_43/no_warmup/sml/SmolLM2-135M-20B-sml-bs4096