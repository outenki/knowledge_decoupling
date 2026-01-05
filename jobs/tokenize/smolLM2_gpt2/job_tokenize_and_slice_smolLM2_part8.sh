#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=48:00:00"
#PJM -L "vnode-core=10"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/tokenize_smolLM2_8.out
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/tokenize_smolLM2_8.out
#PJM -N "tk_sml_8"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/tokenize || exit 1

# part=$1
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash tokenize_and_slice_smolLM2_1024_part.sh 8
