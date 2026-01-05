#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "vnode-core=10"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/tokenize_smolLM2_mn8_1.out
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/tokenize_smolLM2_mn8_1.out
#PJM -N "tk_sm8_1"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/tokenize || exit 1

# tokenizer=$1
# part=$2
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash tokenize_and_slice_smolLM2_1024_mn8.sh "HuggingFaceTB/SmolLM2-1.7B" 1
