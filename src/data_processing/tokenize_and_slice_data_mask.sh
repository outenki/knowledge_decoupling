#!/bin/bash
TOKENIZER=$1
BLOCK_SIZE=4096

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
DATA_NAME=google_re_mix_short
DATA_PATH=/home/pj24001974/ku50001571/projects/knowledge_decoupling/input/evaluate_data/dataset/$DATA_NAME/test
AOA_PATH=$PROJECT_BASE_PATH/data/AOA/aoa.csv
OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/ext/$DATA_NAME-mask-bs$BLOCK_SIZE

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

echo
uv run python $PROJECT_BASE_PATH/src/data_processing/tokenize_and_slice_data.py \
    --tokenizer $TOKENIZER \
    -dp $DATA_PATH \
    -lf local \
    -dc text \
    -aoa $AOA_PATH \
    -s \
    -bs $BLOCK_SIZE \
    -t \
    -o $OUTPUT_PATH

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
