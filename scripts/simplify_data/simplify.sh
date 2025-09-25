#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
DATA_NAME=preprocessed-wikimedia
BASIC_VOCAB=$1
echo "====== simplify with $BASIC_VOCAB======"
uv run python $BASE_PATH/src/simplify_sentence.py \
    -dn $BASE_PATH/data/$DATA_NAME \
    -lf local \
    -bv $BASIC_VOCAB \
    -o $BASE_PATH/data/simplyfied_wikimedia_$BASIC_VOCAB
