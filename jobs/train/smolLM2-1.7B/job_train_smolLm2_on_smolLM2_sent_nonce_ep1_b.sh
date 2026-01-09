#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_sml_on_smolLM2_1.7B_nonce_ep1.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_sml_on_smolLM2_1.7B_nonce_ep1.out
#PJM -N "sln_1.7_e1"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/train/from_init || exit 1

# CONFIG_NAME=$1
# DATA_NAME=$2
# EPOCHS=$3
# DATA_LIMITE=$4
# SUFFIX=${5:-""}
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash train_from_init.sh \
    HuggingFaceTB/SmolLM2-1.7B \
    SmolLM2-1.7B-100B-nonce-SmolLM2-1.7B \
    1 \
    0
