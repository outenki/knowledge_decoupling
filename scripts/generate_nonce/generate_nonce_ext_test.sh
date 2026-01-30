#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-/home/pj25000107/ku50001566/projects/knowledge_decoupling}"
DATA_NAME=$PROJECT_BASE_PATH/data/ext/test/dataset
MAX_N=8

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"
/home/pj25000107/ku50001566/.local/bin/uv run python $PROJECT_BASE_PATH/src/generate_nonce_data.py \
    -d $DATA_NAME \
    -wb $PROJECT_BASE_PATH/data/SmolLM2-1.7B-100B/nonce/vocab/nonce_word_bank.json \
    -lf local \
    -ss text \
    -mn $MAX_N \
    -o $DATA_NAME/nonce/sents_ne/mn_$MAX_N

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
