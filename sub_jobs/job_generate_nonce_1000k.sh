#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=12:00:00"
#PJM -L "gpu=1"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/generate_nonce_1000k.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/generate_nonce_1000k.out
#PJM -N "gen_nonce_1M"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash generate_nonce_data_1000k.sh
