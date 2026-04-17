#!/bin/bash
/home/pj25000107/ku50001566/.local/bin/uv run python /home/pj25000107/ku50001566/projects/knowledge_decoupling/src/preprocess_dataset.py \
    -dn $1 \
    -lf hf \
    -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/preprocessed-$1