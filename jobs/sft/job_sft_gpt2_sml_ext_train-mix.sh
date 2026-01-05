#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_sft_ext_full.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_sft_ext_full.out
#PJM -N "sft_gpt2_ef_c"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/sft || exit 1

# CONFIG_NAME=$1
# INIT_MODEL=$2
# DATA_NAME=$3
# OUTPUT_NAME=$4
# EPOCHS=$5
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash sft.sh \
    gpt2 \
    /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/gpt2/smolLM2/smolLM2-bs1024-dl0-ep1-ext_train-ep1 \
    gpt2/mix \
    smolLM2/smolLM2-bs1024-dl0-ep1-ext_train-ep1-sft_mix_ep3 \
    3
