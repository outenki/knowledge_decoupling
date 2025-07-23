#!/bin/bash
#PJM -L "rscgrp=b-inter"
#PJM -L "elapse=6:00:00"
#PJM -L "gpu=1"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_generate_nonce_data.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_generate_nonce_data.out
#PJM -j

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling || exit 1

singularity exec /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif \
uv run python /home/pj25000107/ku50001566/projects/knowledge_decoupling/src/generate_nonce_data.py \
    -dp wikimedia/wikipedia \
    -dn 20231101.en \
    -lf hf \
    -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/output \
    -l 10000
