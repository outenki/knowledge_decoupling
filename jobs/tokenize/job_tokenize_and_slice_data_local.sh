#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=50:00:00"
#PJM -L "gpu=1"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/slice_smolLM2.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/slice_smolLM2.out
#PJM -N "sl_sml"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/tokenize || exit 1

# DATA_NAME=$1
# DATA_PATH=$BASE_PATH/data/$DATA_NAME
# DATA_COLUMN=$2
# BATCH_SIZE=$3
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash tokenize_and_slice_data_local.sh \
    smolLM2-tokenized \
    text \
    1024
