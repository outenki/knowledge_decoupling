#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_sft_sml.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_sft_sml.out
#PJM -N "sft_sml_e3_c"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/sft || exit 1

# CONFIG_NAME=$1
# INIT_MODEL=$2
# DATA_NAME=$3
# OUTPUT_NAME=$4
# EPOCHS=$5
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash sft.sh \
    HuggingFaceTB/SmolLM2-135M \
    HuggingFaceTB/SmolLM2-135M \
    smollm2/squad_v2_ctxt \
    smollm2-hf \
    3
