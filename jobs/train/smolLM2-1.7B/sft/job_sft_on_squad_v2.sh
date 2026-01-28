#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/2601/job_sft_sml1.7_nonce_squad_v2_balance.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/2601/job_sft_sml1.7_nonce_squad_v2_balance.out
#PJM -N "sft_sml1.7_c"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/sft || exit 1

# CONFIG_NAME=$1
# INIT_MODEL=$2
# DATA_PATH=$3
# OUTPUT_NAME=$4
# EPOCHS=$5
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash sft.sh \
    HuggingFaceTB/SmolLM2-1.7B \
    /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/HuggingFaceTB/SmolLM2-1.7B/SmolLM2-1.7B-100B-nonce-SmolLM2-1.7B-dl0-ep1-tr_0120/checkpoint-20672 \
    /home/pj25000107/ku50001566/projects/knowledge_decoupling/input/tokenized/SmolLM2-1.7B/sft/squad_v2_ctxt_ans \
    /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/SmolLM2-1.7B/nonce/SmolLM2-1.7B-100B-nonce-SmolLM2-1.7B-dl0-ep1-checkpoint-20672-sft-squad_v2_ctxt_ans-e3 \
    3
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash sft.sh \
    HuggingFaceTB/SmolLM2-1.7B \
    /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/HuggingFaceTB/SmolLM2-1.7B/SmolLM2-1.7B-100B-nonce-SmolLM2-1.7B-dl0-ep1-tr_0120/checkpoint-20672 \
    /home/pj25000107/ku50001566/projects/knowledge_decoupling/input/tokenized/SmolLM2-1.7B/sft/squad_v2_ctxt_1000_unans \
    /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/SmolLM2-1.7B/nonce/SmolLM2-1.7B-100B-nonce-SmolLM2-1.7B-dl0-ep1-checkpoint-20672-sft-squad_v2_ctxt_1000_unans-e3 \
    3