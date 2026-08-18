#!/bin/bash
#PBS -q lg
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/blimp.log
#PBS -N blimp_olmo_core


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/allenai/OLMo-2-0425-1B/SmolLM2-135M-20B-core-bs1024
sh lm_eval_blimp.sh $MODEL_PATH