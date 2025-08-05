
#!/bin/bash
uv run python /home/pj25000107/ku50001566/projects/knowledge_decoupling/src/preprocess_dataset.py \
    -dp wikitext \
    -lf hf \
    -o data/preprocessed_1000k \
    -l 1000000