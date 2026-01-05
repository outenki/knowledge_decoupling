#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
OUTPUT_PATH=$BASE_PATH/data/SmolLM2
DATA_NAME=$1
TOKENIZER=$2

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

echo
echo "====== preprocess part$part ======"
/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/tokenize_and_slice_data.py \
    --tokenizer $TOKENIZER \
    -dn $DATA_NAME \
    -lf hf -dc text -t -s -bs 1024 \
    -o $OUTPUT_PATH/tokenized/$TOKENIZER/smollm2-tokenized_bs1024

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
