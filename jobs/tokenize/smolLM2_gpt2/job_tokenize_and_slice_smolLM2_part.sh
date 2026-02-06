#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=10:00:00"
#PJM -L "vnode-core=10"
#PJM -N "tk_sml"
#PJM -e ./logs/%n_%j_%J.err
#PJM -o ./logs/%n_%j_%J.out

# module load singularity-ce

cd $HOME/projects/knowledge_decoupling/scripts/tokenize || exit 1

/bin/bash tokenize_smolLM2_1024_part.sh gpt2 ${PJM_BULKNUM:-0}
