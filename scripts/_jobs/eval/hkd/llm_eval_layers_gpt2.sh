#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/gpt2.log


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_NAME="openai-community/gpt2"
sh llm_eval_layers.sh $MODEL_NAME arc_easy,arc_challenge,commonsense_qa 1 12 1
