#!/bin/bash
#PBS -q lg
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/smolLM2-135M-nonce.log


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/HuggingFaceTB/SmolLM2-135M/SmolLM2-135M-20B-nonce-bs1024
sh llm_eval.sh $MODEL_PATH