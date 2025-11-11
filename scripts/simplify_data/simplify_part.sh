#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
DATA_NAME=SmolLM2
BASIC_VOCAB=$1
PART=$2
SIZE=100000


start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"


echo
echo "====== simplify with $BASIC_VOCAB ======"
echo "====== preprocess $part ======"
/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/simplify_sentence.py \
    -dn $DATA_NAME \
    -lf hf \
    -sf $(($PART * $SIZE)) \
    -sk source \
    -sv stack_edu infimm_webmath \
    -l $SIZE \
    -bv $BASIC_VOCAB \
    -o $BASE_PATH/data/simplyfied_SmolLM2_$BASIC_VOCAB/nonce-parts/part$part

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
