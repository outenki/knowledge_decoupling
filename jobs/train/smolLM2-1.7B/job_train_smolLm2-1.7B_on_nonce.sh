#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=50:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_sml_on_smolLM2_1.7B_nonce_ep1.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_sml_on_smolLM2_1.7B_nonce_ep1.out
#PJM -N "sln_1.7_e1"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/train/continue/smolLM2-1.7B || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash continue_train_smollm2-1.7B_nonce.sh
