#!/bin/bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${PROJECT_BASE_PATH:-$project_root}"
python=.venv/bin/python
model_name=meta-llama/Llama-3.2-1B
model_root="output/$model_name/seed_42/no_warmup"
sml_tokenizer="${SML_TOKENIZER:-$model_root/sml/SmolLM2-135M-20B-sml-bs4096}"
rnd_id_tokenizer="${RND_ID_TOKENIZER:-$model_root/sml_rnd_id/SmolLM2-135M-20B-rnd_id-bs4096}"
max_length="${CONTEXT_MAX_LENGTH:-4096}"
tokenized_root="input/tokenized/$model_name/ext"

echo "Splitting evaluation contexts and no-context QA"
contexts=$("$python" -m scripts.data_processing.prepare_context_qa "$@")

export HF_HUB_OFFLINE=1
while IFS=$'\t' read -r task context_file; do
    case "$task" in
        *_rnd_id) tokenizer="$rnd_id_tokenizer" ;;
        *) tokenizer="$sml_tokenizer" ;;
    esac

    if [[ ! -f "$tokenizer/tokenizer.json" ]]; then
        echo "Missing checkpoint tokenizer: $tokenizer/tokenizer.json" >&2
        exit 1
    fi

    echo "Tokenizing $context_file with $tokenizer"
    "$python" -m scripts.data_processing.tokenize_dataset_from_json \
        --only-context \
        --keep-tokenizer-vocab \
        --max-length "$max_length" \
        --tokenizer "$tokenizer" \
        --input-path "$context_file" \
        --output-path "$tokenized_root/${task}_context-bs${max_length}"
done <<< "$contexts"
