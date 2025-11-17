#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_sft_gpt2_c.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_sft_gpt2_c.out
#PJM -N "sft_gpt2_e3_c"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/train_large/sft || exit 1

# CONFIG_NAME=$1
# INIT_MODEL=$2
# DATA_NAME=$3
# EPOCHS=$4
# DATA_LIMITE=$5
# SUFFIX=${6:-""}
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash sft_gpt.sh \
    gpt-large \
    gpt2 \
    squad_v2_ctxt \
    3 \
    0 \
    gpt2
