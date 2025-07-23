#!/bin/bash
uv run python /home/pj25000107/ku50001566/projects/knowledge_decoupling/src/generate_nonce_data.py \
    -dp wikimedia/wikipedia \
    -dn 20231101.en \
    -lf hf \
    -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/output \
    -l 10000