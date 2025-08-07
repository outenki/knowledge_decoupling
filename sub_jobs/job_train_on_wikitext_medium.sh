#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=12:00:00"
#PJM -L "gpu=1"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_wikitext_gpt_medium.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_wikitext_gpt_medium.out
#PJM -N "wikitext_m_train"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash train_on_wikitext_gpt_medium.sh
