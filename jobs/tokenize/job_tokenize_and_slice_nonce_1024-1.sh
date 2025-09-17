#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=20:00:00"
#PJM -L "gpu=1"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/tokenize_nonce_1024_1.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/tokenize_nonce_1024_1.out
#PJM -N "tk_nonce_1024_1"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash tokenize_and_slice_nonce_1024.sh merged-1
