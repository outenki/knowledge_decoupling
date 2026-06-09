PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
# DATA_LIMIT=1000
# DATA_LIMIT=2000000
DATA_LIMIT=1000000
DATA_NAME="SmolLM2-20B"
DATA_PATH=$PROJECT_BASE_PATH/data/
START_FROM=$((${1:-0} * DATA_LIMIT))
THRESHOLD=2
echo "Filtering dataset with threshold: $THRESHOLD, starting from: $START_FROM"
uv run python $PROJECT_BASE_PATH/src/data_processing/filter_dataset.py \
    --dataset-name "EleutherAI/SmolLM2-135M-20B" \
    --dataset-split "train" \
    --model-name "en_core_web_sm" \
    --text-column "text" \
    --filter-key "entry_count" \
    --threshold $THRESHOLD \
    --batch-size 512 \
    --num-proc 4 \
    --start-from $START_FROM \
    --data-limit $DATA_LIMIT \
    --token-frequency-path "$DATA_PATH/token_frequency/en_core_web_sm/token_frequency.json" \
    --kept-indices-path $DATA_PATH/$DATA_NAME/kept_indices.json \
    --output-path "$DATA_PATH/filtered_dataset/smolLM2-135M-20B/replaced/threshold_$THRESHOLD/part_$1"
