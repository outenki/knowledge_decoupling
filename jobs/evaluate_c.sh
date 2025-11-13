#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=12:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_evaluation_c.out
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_evaluation_c.out
#PJM -N "eval_c"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash evaluate.sh
