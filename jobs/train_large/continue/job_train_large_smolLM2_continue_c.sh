#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_large_smolLM2_continue_c_e1.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_large_smolLM2_continue_c_e1.out
#PJM -N "sml_e1"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/train_large/continue || exit 1

# CONFIG_NAME=$1
# DATA_NAME=$2
# CHECKPOINT=$3
# EPOCHS=$4
# DATA_LIMITE=$5
# SUFFIX=$6
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif \
    /bin/bash train_gpt_continue.sh \
        gpt-large \
        smolLM2-bs1024 \
        /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/gpt-large/smolLM2-bs1024-dl0-ep1-tr/checkpoint-8514 \
        1 \
        0
