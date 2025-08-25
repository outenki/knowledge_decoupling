#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling

# 0821
# tokenize
# DATA_NAME=wikimedia
# /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/tokenize_and_slice_data.py \
#     -dn $DATA_NAME -lf hf -dc text -t -o $BASE_PATH/data/$DATA_NAME

# slice
DATA_NAME=wikimedia
/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/tokenize_and_slice_data.py \
    -dn $BASE_PATH/data/wikimedia_tokenized -lf local -dc text -s -o $BASE_PATH/input/$DATA_NAME

# 0820
# DATA_NAME=wikitext-103
# /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/tokenize_and_slice_data.py \
#     -dn $DATA_NAME -lf hf -dc text -t -s -o $BASE_PATH/input/$DATA_NAME