#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "vnode-core=50"
#PJM -L "jobenv=singularity"
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/generate_nonce_1.out
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/generate_nonce_1.err
#PJM -N "nonce_1"


module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash generate_nonce_wikimedia.sh 1
