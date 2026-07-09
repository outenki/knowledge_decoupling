#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/gpt2.log


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/"openai-community/gpt2"
L1=1
L2=12
STEP=1
sh llm_eval_layers_qa.sh $MODEL_NAME $L1 $L2 $STEP
sh llm_eval_layers_blimp.sh $MODEL_NAME $L1 $L2 $STEP