#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=70:00:00"
#PJM -L "gpu=4"
#PJM -e logs/context_qa.log
#PJM -o logs/context_qa.log
#PJM -N "eval_llama_cqa"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/meta-llama/Llama-3.2-1B/SmolLM2-135M-20B-core_ent-bs4096
sh llm_eval_context_qa.sh $MODEL_PATH