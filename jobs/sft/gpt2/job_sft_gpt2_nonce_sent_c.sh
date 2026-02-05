#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=10:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_sft_gpt2_nonce_sent.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_sft_gpt2_nonce_sent.out
#PJM -N "sft_gn_e3_c"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/train/sft/gpt2 || exit 1

# CONFIG_NAME=$1
# INIT_MODEL=$2
# DATA_NAME=$3
# OUTPUT_NAME=$4
# EPOCHS=$5
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash ./sft_gpt2_nonce_squad.sh
