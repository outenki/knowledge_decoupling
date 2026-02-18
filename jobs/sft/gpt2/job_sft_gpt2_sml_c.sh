#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=4:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -e /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_sft_sml.err
#PJM -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/logs/job_sft_sml.out
#PJM -N "sft_gs_e3_c"

# module load singularity-ce

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/train/sft/gpt2 || exit 1

/bin/bash ./sft_gpt2_sml_squad.sh