#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_gpt_large_on_smolLM2_nonce_ep1_c.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_gpt_large_on_smolLM2_nonce_ep1_c.out
#PJM -N "c_smn_e1"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/train_large/from_init || exit 1

# CONFIG_NAME=$1
# DATA_NAME=$2
# EPOCHS=$3
# DATA_LIMITE=$4
# SUFFIX=$5
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash train_gpt_from_init.sh \
    gpt-large \
    smolLM2-nonce-mn3-bs1024 \
    1 \
    0
