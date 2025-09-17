#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=12:00:00"
#PJM -L "gpu=1"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/merge_data.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/merge_data.out
#PJM -N "merge_data"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash merge_data.sh
