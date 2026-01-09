#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_evaluate_gpt2_sft_c.out
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_evaluate_gpt2_sft_c.err
#PJM -N "evl_gpt2_sft_c"

module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/evaluate/gpt2_sft || exit 1

# singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash evaluate_on_options_0shots.sh
singularity exec --nv /home/pj25000107/ku50001566/nlp-singularity/nlp-singularity.sif /bin/bash evaluate_on_generation_0shots.sh
