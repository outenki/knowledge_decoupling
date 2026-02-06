#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=10:00:00"
#PJM -L "vnode-core=10"
#PJM -L "jobenv=singularity"
#PJM -e ./logs/tokenize_smolLM2_%j.err
#PJM -o ./logs/tokenize_smolLM2_%j.out
#PJM -N "tk_sml_0"

# module load singularity-ce

cd $HOME/projects/knowledge_decoupling/scripts/tokenize || exit 1
mkdir -p ./logs

/bin/bash tokenize_smolLM2_1024_part.sh gpt2 ${PJM_BULKNUM:-0}
