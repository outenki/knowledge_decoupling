#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_sft_gpt2_ext_nonce_ne_balanced.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_sft_gpt2_ext_nonce_ne_balanced.out
#PJM -N "sft_g_ene_c"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/sft || exit 1

# CONFIG_NAME=$1
# INIT_MODEL=$2
# DATA_PATH=$3
# OUTPUT_NAME=$4
# EPOCHS=$5
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash sft.sh \
    gpt2 \
    /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/gpt2/nonce/smolLM2-nonce-bs1024-dl0-ep1-ext_test_nonce_ne-ep3 \
    /home/pj25000107/ku50001566/projects/knowledge_decoupling/input/tokenized/gpt2/sft/squad_v2_ctxt_ans \
    /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/gpt2/nonce/smolLM2-nonce-bs1024-dl0-ep1-ext_test-nonce_ne-ep3-sft-squad_v2_ctxt_ans-ep3 \
    3

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash sft.sh \
    gpt2 \
    /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/gpt2/nonce/smolLM2-nonce-bs1024-dl0-ep1-ext_test_nonce_ne-ep3 \
    /home/pj25000107/ku50001566/projects/knowledge_decoupling/input/tokenized/gpt2/sft/squad_v2_ctxt_1000_unans \
    /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/gpt2/nonce/smolLM2-nonce-bs1024-dl0-ep1-ext_test-nonce_ne-ep3-sft-squad_v2_ctxt_1000_unans-ep3 \
    3