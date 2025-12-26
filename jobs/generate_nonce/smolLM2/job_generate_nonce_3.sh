#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "vnode-core=10"
#PJM -L "jobenv=singularity"
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/generate_nonce_smolLM2_3.out
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/generate_nonce_smolLM2_3.out
#PJM -N "gn_sml_3"


module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/generate_nonce || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash generate_nonce_smolLM2.sh 3
