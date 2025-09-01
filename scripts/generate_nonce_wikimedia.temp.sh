#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
DATA_NAME=preprocessed-wikimedia
ITER_NUM=122
SIZE=100000
START=$(($1 * $SIZE * $ITER_NUM))
END=$(($(($1 + 1)) * $SIZE * $ITER_NUM -1))

for i in $(seq $START $SIZE $END)
do
    part=$(($i / $SIZE))
    end_part=$(($END / $SIZE))
    if (( $part > 852 )); then
        echo
        echo "====== preprocess $part / $end_part ======"
        echo "start at $(date)"
        /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/generate_nonce_data.py \
            -dn $BASE_PATH/data/$DATA_NAME \
            -lf local \
            -o $BASE_PATH/data/wikimedia-nonce/part$part \
            -sf $i \
            -l $SIZE
        echo "end at $(date)"
    fi
done

# echo
# echo "====== mine ======"
# echo "start at $(date)"
# /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/generate_nonce_data.py \
#     -dn $BASE_PATH/data/$DATA_NAME \
#     -lf local \
#     -o $BASE_PATH/data/${DATA_NAME}-nonce \
#     -sf 200000 \
#     -l 100000
# echo "end at $(date)"
