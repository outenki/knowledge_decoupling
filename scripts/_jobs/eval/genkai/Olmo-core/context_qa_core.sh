#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=70:00:00"
#PJM -L "gpu=4"
#PJM -e logs/context_qa_core.log
#PJM -o logs/context_qa_core.log
#PJM -N "eval_olmo_cqc"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_PATH=$PROJECT_BASE_PATH/output/allenai/OLMo-2-0425-1B/SmolLM2-135M-20B-core_ent-bs4096
sh llm_eval_context_qa.sh $MODEL_PATH