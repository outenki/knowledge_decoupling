#!/bin/bash
PART=$1

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
DATA_PATH=$PROJECT_BASE_PATH/data/SmolLM2-135M-20B/core_ent
SIZE=1800000
START=$(($PART * $SIZE))
END=$(($(($PART + 1)) * $SIZE -1))

start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"

# echo
# echo "====== Generating core data part$PART (from $START to $END) ======"
PART_PATH=$DATA_PATH/parts/part_$PART
# uv run python $PROJECT_BASE_PATH/src/data_processing/core_data/generate_core_data.py \
#     -d SmolLM2-20B \
#     -ki $PROJECT_BASE_PATH/data/SmolLM2-135M-20B/kept_indices.json \
#     -lf hf \
#     --start-from $START \
#     --limit $SIZE \
#     -aoa $PROJECT_BASE_PATH/data/AOA/aoa.csv \
#     -at 10 \
#     --multi-process \
#     -o $PART_PATH \
#     -sp train \
#     --core-delimiter "<>" \
#     --ent-generator "ENT" \
#     --unk-generator "UNK"

BLOCK_SIZE=4096
for TOKENIZER in  meta-llama/Llama-3.2-1B allenai/OLMo-2-0425-1B Qwen/Qwen2.5-0.5B
do
    echo ">>> Tokenizing with $TOKENIZER"
    TOKENIZER_PATH=$DATA_PATH/tokenized/$TOKENIZER/tokenizer
    echo ">>> Loading tokenizer from $TOKENIZER_PATH"
    OUTPUT_PATH=$DATA_PATH/tokenized/$TOKENIZER/part_$PART

    uv run python $PROJECT_BASE_PATH/src/data_processing/tokenize_and_slice_data.py \
        --tokenizer $TOKENIZER_PATH \
        -dp $PART_PATH \
        -dc "core" \
        -lf local \
        -sp train \
        -s \
        -bs $BLOCK_SIZE \
        -t \
        -o $OUTPUT_PATH
done

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
