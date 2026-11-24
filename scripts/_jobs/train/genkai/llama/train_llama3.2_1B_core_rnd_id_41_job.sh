#!/bin/bash
#PJM -L "rscgrp=b-batch"
#PJM -L "elapse=100:00:00"
#PJM -L "gpu=4"
#PJM -e logs/train_Llama3.2_1B_core_rnd_id_41.log
#PJM -o logs/train_Llama3.2_1B_core_rnd_id_41.log
#PJM -N "tr_41_lcri"



SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/../../common/run_llama3_2_1b.sh" Llama3.2_1B-sml_rnd_id 41
