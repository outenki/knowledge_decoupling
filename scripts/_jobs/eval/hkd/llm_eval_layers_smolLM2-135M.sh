#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/SmolLM2-135M.log


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/"HuggingFaceTB/SmolLM2-135M"
L1=2
L2=30
STEP=2

sh llm_eval_layers_qa.sh $MODEL_PATH $L1 $L2 $STEP
sh llm_eval_layers_blimp.sh $MODEL_PATH $L1 $L2 $STEP