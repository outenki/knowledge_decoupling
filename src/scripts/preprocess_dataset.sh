
#!/bin/bash
uv run python /home/pj25000107/ku50001566/projects/knowledge_decoupling/src/preprocess_dataset.py \
    -dp wikimedia/wikipedia \
    -dn 20231101.en \
    -lf hf \
    -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/output/data/preprocessed \
    -l 10000