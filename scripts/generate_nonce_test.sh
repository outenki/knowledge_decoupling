#!/bin/bash
BASE_PATH=/Users/ou/Developer/projects/knowledge_decoupling
DATA_NAME=preprocessed-wikimedia

# echo "====== gpt ======"
# echo "start at $(date)"
# python $BASE_PATH/src/temp.py \
#     -dp $BASE_PATH/data/$DATA_NAME \
#     -lf local \
#     -o $BASE_PATH/data/${DATA_NAME}-gpt-nonce
# echo "end at $(date)"

echo "\n====== mine ======"
echo "start at $(date)"
python $BASE_PATH/src/generate_nonce_data.py \
    -dn $BASE_PATH/data/$DATA_NAME \
    -lf local \
    -o $BASE_PATH/data/${DATA_NAME}-mine-nonce \
    -l 100000
echo "end at $(date)"

# echo "\n====== old ======"
# echo "start at $(date)"
# python $BASE_PATH/src/old.py \
#     -dp $BASE_PATH/data/$DATA_NAME \
#     -lf local \
#     -o $BASE_PATH/data/${DATA_NAME}-old-nonce
# echo "end at $(date)"
