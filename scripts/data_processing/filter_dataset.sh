for THRESHOLD in 1 2 3 4 5; do
    echo "Filtering dataset with threshold: $THRESHOLD"
    uv run python filter_dataset.py \
        --dataset-name "EleutherAI/SmolLM2-135M-20B" \
        --dataset-split "train" \
        --model-name "en_core_web_sm" \
        --text-column "text" \
        --filter-key "entry_count" \
        --threshold $THRESHOLD \
        --batch-size 512 \
        --num-proc 4 \
        --token-frequency-path "../../data/token_frequency/en_core_web_sm/token_frequency.json" \
        --kept-indices-path "../../data/filtered_dataset/smolLM2-135M-20B/kept_indices.json" \
        --output-path "../../data/filtered_dataset/smolLM2-135M-20B/skip_oov/threshold_$THRESHOLD"
done
