#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_continue_wiki_nonce_size_3.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_continue_wiki_nonce_size_3.out
#PJM -N "wk_ns_e3"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/train/continue || exit 1

# CONFIG_NAME=$1
# DATA_NAME=$2
# CHECKPOINT=$3
# EPOCHS=$4
# DATA_LIMITE=$5
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif \
    /bin/bash train_gpt_continue.sh \
        gpt2 \
        wikimedia-bs1024 \
        /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/gpt2/wikimedia-bs1024-dl1_020_000-ep2/checkpoint-7486 \
        3 \
        1_020_000
