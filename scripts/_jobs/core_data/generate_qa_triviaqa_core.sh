#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "elapse=50:00:00"
#PJM -L "vnode-core=10"
#PJM -e logs/generate_qa_triviaqa_core.out
#PJM -o logs/generate_qa_triviaqa_core.out
#PJM -N "gen_triviaqa_core"


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/data_processing

sh ./generate_qa_triviaqa_core.sh
