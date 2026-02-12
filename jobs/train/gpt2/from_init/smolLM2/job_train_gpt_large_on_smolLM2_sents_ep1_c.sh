#!/bin/bash
#PJM -L "rscgrp=c-batch"
#PJM -L "elapse=24:00:00"
#PJM -L "gpu=4"
#PJM -N "gpt_sml_sent"
#PJM -e ./logs/%n_%j.err
#PJM -o ./logs/%n_%j.out

cd $HOME/projects/knowledge_decoupling/scripts/train/from_init/gpt2 || exit 1

/bin/bash sml_sents.sh
