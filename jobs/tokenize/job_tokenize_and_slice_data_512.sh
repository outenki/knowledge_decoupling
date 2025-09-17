#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=20:00:00"
#PJM -L "gpu=1"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/tokenize_data_512.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/tokenize_data_512.out
#PJM -N "tokenize_512"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash tokenize_and_slice_data_512.sh
