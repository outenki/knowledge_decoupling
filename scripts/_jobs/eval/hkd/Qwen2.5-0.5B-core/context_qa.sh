#!/bin/bash
#PBS -q lg
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/context_qa.log
#PBS -N cqa_qwen_core


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/Qwen/Qwen2.5-0.5B/SmolLM2-135M-20B-core-bs1024
sh llm_eval_context_qa.sh $MODEL_PATH