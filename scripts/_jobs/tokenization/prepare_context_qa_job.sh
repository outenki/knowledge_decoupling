#!/bin/bash
#PJM -L "rscgrp=a-batch"
#PJM -L "vnode-core=10"
#PJM -L "elapse=02:00:00"
#PJM -N "context_qa"
#PJM -j

set -euo pipefail

cd "${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
exec bash scripts/data_processing/prepare_context_qa.sh "$@"
