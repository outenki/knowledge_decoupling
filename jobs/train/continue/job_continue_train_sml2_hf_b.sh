#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job-continue-train-gpt_hf-b.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job-continue-train-gpt_hf-b.out
#PJM -N "ct_gpt_hf_b"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/train/continue/ext-train || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash continue_train_smollm2_hf_ext-train.sh
