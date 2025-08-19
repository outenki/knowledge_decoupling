#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=12:00:00"
#PJM -L "gpu=1"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_preprocess_data_10m.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_preprocess_data_10m.out
#PJM -N "prep_data_10m"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash preprocess_dataset_10m.sh
