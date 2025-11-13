#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=48:00:00"
#PJM -L "vnode-core=10"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_tokenize_and_slice_simplified_ox_part2.out
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_tokenize_and_slice_simplified_ox_part2.out
#PJM -N "tk_ox_2"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/tokenize || exit 1

# part=$1
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash tokenize_and_slice_data_local.sh 2
