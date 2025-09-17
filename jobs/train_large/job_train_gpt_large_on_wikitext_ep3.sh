#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=12:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_gpt_large_on_wikitext_ep3.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_train_gpt_large_on_wikitext_ep3.out
#PJM -N "tlw_ep3"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash train_gpt_large_on_wikitext_ep3.sh
