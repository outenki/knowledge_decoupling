#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling


# # ************** 0808 *************
# # ** wikitext raw data (long)    **
/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/tokenize_data.py \
    -dn wikimedia -lf hf -dc text -bs all -o $BASE_PATH/input/wikimedia
