#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_evaluate_on_sml3-3B_options_0shot_b.out
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_evaluate_on_sml3-3B_options_0shot_b.out
#PJM -N "eo_sml3_b"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/evaluate/smolLM3-3B || exit 1

singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash evaluate_on_options_0shots.sh
