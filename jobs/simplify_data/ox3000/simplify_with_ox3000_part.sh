#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=48:00:00"
#PJM -L "vnode-core=10"
#PJM -L "jobenv=singularity"
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/spl_ox_part.out
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/spl_ox_part.out
#PJM -N "spl_ox_p"


module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/simplify_data || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash simplify_part.sh ox3000
