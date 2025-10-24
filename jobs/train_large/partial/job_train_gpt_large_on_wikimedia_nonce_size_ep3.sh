#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_gpt_large_on_wikitext_nonce_size_ep3.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_gpt_large_on_wikitext_nonce_size_ep3.out
#PJM -N "wk_ns_e3"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/train_large/from_init || exit 1

# CONFIG_NAME=$1
# DATA_NAME=$2
# EPOCHS=$3
# DATA_LIMITE=${4:-0}
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash train_gpt_from_init.sh \
    gpt-large \
    wikimedia-bs1024 \
    3 \
    1_020_000
