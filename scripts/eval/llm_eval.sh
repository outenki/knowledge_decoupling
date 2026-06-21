#!/bin/bash
MODEL_PATH=$1

# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

sh llm_eval_blimp.sh $MODEL_PATH
sh llm_eval_qa.sh $MODEL_PATH
sh llm_eval_context_qa.sh $MODEL_PATH