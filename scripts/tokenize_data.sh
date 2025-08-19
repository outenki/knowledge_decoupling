#!/bin/bash
SCRIPT_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/src


# # ****** 0802 ******
# # **  nonce data  **
echo "==== tokenizing wikitext ======"
for l in 128 512;do
    /home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/tokenize_data.py \
        -dp $SCRIPT_PATH/../data/preprocessed_100k \
        -lf local \
        -dc text \
        -dml $l \
        -o ../input/wikitext_$l

    echo "==== tokenizing nonce ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/tokenize_data.py \
        -dp $SCRIPT_PATH/../data/wikitext_with_nonce \
        -lf local \
        -dc nonce \
        -dml $l \
        -o ../input/nonce_$l
done


# # ************** 0808 *************
# # ** wikitext raw data (long)    **
# echo "==== tokenizing wikitext raw ======"
# /home/pj25000107/ku50001566/.local/bin/uv run python $SCRIPT_PATH/tokenize_data.py \
#     -dp wikitext \
#     -lf hf \
#     -dc text \
#     -dml 1024 \
#     -o ../input/wikitext-raw
