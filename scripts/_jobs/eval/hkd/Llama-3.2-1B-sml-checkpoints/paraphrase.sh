#!/bin/bash
#PBS -q c30897g
#PBS -l select=1:ngpus=4
#PBS -l walltime=50:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/paraphrase.log
#PBS -N evsml_para


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/no_warmup/sml
sh lm_eval_paraphrase.sh $MODEL_PATH/SmolLM2-135M-20B-sml-bs4096
sh lm_eval_paraphrase.sh $MODEL_PATH/checkpoints/check_1
sh lm_eval_paraphrase.sh $MODEL_PATH/checkpoints/check_3
sh lm_eval_paraphrase.sh $MODEL_PATH/checkpoints/check_5
sh lm_eval_paraphrase.sh $MODEL_PATH/checkpoints/check_7
sh lm_eval_paraphrase.sh $MODEL_PATH/checkpoints/check_9