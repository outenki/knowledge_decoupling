#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
DATA_PATH=$BASE_PATH/data/SmolLM2/1027/mn_8
OUTPUT_PATH=$BASE_PATH/data/SmolLM2/tokenized_bs1024
ITER_NUM=10
SIZE=100000
START=$(($1 * $SIZE * $ITER_NUM))
END=$(($(($1 + 1)) * $SIZE * $ITER_NUM -1))

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

for i in $(seq $START $SIZE $END)
do
    part=$(($i / $SIZE))
    echo
    echo "====== preprocess part$part ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/tokenize_and_slice_data.py \
        -dn $DATA_PATH/part$part -lf local -dc text -t -s -bs 1024 -o $OUTPUT_PATH-part$part
done

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
