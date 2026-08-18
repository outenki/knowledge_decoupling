#!/bin/bash
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

sh lm_eval_blimp.sh $MODEL_PATH
sh lm_eval_qa.sh $MODEL_PATH
sh lm_eval_context_qa.sh $MODEL_PATH