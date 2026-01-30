#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_gpt_large_on_smolLM2_qa_test_ep1_c.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_gpt_large_on_smolLM2_qa_test_ep1_c.out
#PJM -N "tr_gsm_c"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/train/from_init || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash train_from_init-mix.sh gpt2 1
