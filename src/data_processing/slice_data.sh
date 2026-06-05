PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
TOKENIZER="Qwen/Qwen3.5-0.8B-Base"
INPUT_DATA_NAME=$1
INPUT_DATA_PATH="$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/train/$INPUT_DATA_NAME-bs4096/"
BS=1024
OUTPUT_DATA_NAME="$INPUT_DATA_NAME-bs$BS"
OUTPUT_PATH="$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/train/$OUTPUT_DATA_NAME/"

uv run python tokenize_and_slice_data.py \
    --tokenizer $TOKENIZER \
    --data-path $INPUT_DATA_PATH \
    -np 16 \
    -sp train \
    -lf local \
    -s \
    -bs $BS \
    -o $OUTPUT_PATH