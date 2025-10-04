#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=50:00:00"
#PJM -L "vnode-core=50"
#PJM -L "jobenv=singularity"
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/spl_ox3000.out
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/spl_ox3000.err
#PJM -N "spl_850"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/simplify_data || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash simplify.sh ox3000
