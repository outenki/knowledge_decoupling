#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_gpt_nonce_on_ext_test.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_gpt_nonce_on_ext_test.out
#PJM -N "sml_e1"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/train/continue/ext-test || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash continue_train_gpt2_nonce_ext-test.sh
