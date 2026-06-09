#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/SmolLM2-360M.log


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_NAME="HuggingFaceTB/SmolLM2-360M"
sh llm_eval_layers_qa.sh $MODEL_NAME 2 32 2