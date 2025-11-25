#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_sft_ox3000_c.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_sft_ox3000_c.out
#PJM -N "sft_ox_e3_c"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/sft || exit 1

# CONFIG_NAME=$1
# INIT_MODEL=$2
# DATA_NAME=$3
# OUTPUT_NAME=$4
# EPOCHS=$5
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash sft.sh \
    gpt2 \
    /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/gpt2/smolLM2/smolLM2-ox3000-bs1024-dl0-ep3 \
    mix \
    smolLM2-ox3000-bs1024-dl0-ep3 \
    3 \
