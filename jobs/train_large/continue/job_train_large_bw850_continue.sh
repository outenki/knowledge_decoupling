#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_continue_bw850_ep3-10.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_continue_bw850_ep3-10.out
#PJM -N "tl_bw_ep10"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/train_large/continue || exit 1

# CONFIG_NAME=$1
# DATA_NAME=$2
# CHECKPOINT=$3
# EPOCHS=$4
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif \
    /bin/bash train_gpt_continue.sh \
        gpt-large \
        simplyfied-wikimedia-bw850-bs1024 \
        /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/gpt-large/simplyfied-wikimedia-bw850-bs1024-ep3 \
        10
