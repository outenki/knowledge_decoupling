#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_gpt_large_on_wikimedia_nonce_full.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_gpt_large_on_wikitmedia_nonce_full.out
#PJM -N "tl_wn_fl"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash train_gpt_large_on_wikimedia_nonce_full.sh
