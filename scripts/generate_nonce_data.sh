#!/bin/bash
echo "====== wikitext_with_nonce_1000k ======"
/home/pj25000107/ku50001566/.local/bin/uv run python /home/pj25000107/ku50001566/projects/knowledge_decoupling/src/generate_nonce_data.py \
    -dp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/preprocessed \
    -lf local \
    -l 1000000 \
    -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/wikitext_with_nonce_1000k

echo "====== wikitext_with_nonce_500k ======"
/home/pj25000107/ku50001566/.local/bin/uv run python /home/pj25000107/ku50001566/projects/knowledge_decoupling/src/generate_nonce_data.py \
    -dp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/preprocessed \
    -lf local \
    -l 500000 \
    -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/wikitext_with_nonce_500k

echo "====== wikitext_with_nonce_100k ======"
/home/pj25000107/ku50001566/.local/bin/uv run python /home/pj25000107/ku50001566/projects/knowledge_decoupling/src/generate_nonce_data.py \
    -dp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/preprocessed \
    -lf local \
    -l 100000 \
    -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/wikitext_with_nonce_100k

echo "====== wikitext_with_nonce_50k ======"
/home/pj25000107/ku50001566/.local/bin/uv run python /home/pj25000107/ku50001566/projects/knowledge_decoupling/src/generate_nonce_data.py \
    -dp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/preprocessed \
    -lf local \
    -l 100000 \
    -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/wikitext_with_nonce_50k

echo "====== wikitext_with_nonce_10k ======"
/home/pj25000107/ku50001566/.local/bin/uv run python /home/pj25000107/ku50001566/projects/knowledge_decoupling/src/generate_nonce_data.py \
    -dp /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/preprocessed \
    -lf local \
    -l 100000 \
    -o /home/pj25000107/ku50001566/projects/knowledge_decoupling/data/wikitext_with_nonce_10k