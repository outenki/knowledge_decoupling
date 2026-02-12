#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=4:00:00"
#PJM -L "gpu=4"
#PJM -L "jobenv=singularity"
#PJM -N "gpt2_shuffled_sents_ext_test"
#PJM -e ./logs/%n_%j_%J.err
#PJM -o ./logs/%n_%j_%J.out

cd /home/pj25000107/ku50001566/projects/knowledge_decoupling/scripts/train/ext/ext-test || exit 1

/bin/bash continue_train_gpt2_shuffled_sents-ext_test_wo_answer.sh
