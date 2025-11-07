#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
DATA_NAME=SmolLM2
BASIC_VOCAB=$1
ITER_NUM=10
SIZE=100000
START=$(($2 * $SIZE * $ITER_NUM))
END=$(($(($2 + 1)) * $SIZE * $ITER_NUM -1))
MAX_N=3


start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

for i in $(seq $START $SIZE $END)
do
    part=$(($i / $SIZE))
    echo
    echo "====== simplify with $BASIC_VOCAB ======"
    echo "====== preprocess $part ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/simplify_sentence.py \
        -dn $DATA_NAME \
        -lf hf \
        -sf $i \
        -sk source \
        -sv stack_edu infimm_webmath \
        -l $SIZE \
        -bv $BASIC_VOCAB \
        -o $BASE_PATH/data/simplyfied_SmolLM2_$BASIC_VOCAB/nonce-parts/part$part
done

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
