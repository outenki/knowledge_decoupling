#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/qa.log
#PBS -N qa_qwen_sml


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/Qwen/Qwen2.5-0.5B/SmolLM2-135M-20B-bs1024
sh lm_eval_qa.sh $MODEL_PATH