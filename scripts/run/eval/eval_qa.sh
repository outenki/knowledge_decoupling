#!/bin/bash
start_time=$(date +"%s")
echo "start time: $(date -d @"$start_time" +"%D %T")"

module load cuda/13.2.0
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            if [[ -n "$2" && "$2" != -* ]]; then
                CONFIG_NAME="$2"
                shift 2
            else
                echo "err: -c | --config need a value"
                exit 1
            fi
            ;;
        -p|--model-path)
            if [[ -n "$2" && "$2" != -* ]]; then
                MODEL_PATH="$2"
                shift 2
            else
                echo "err: -p | --model-path need a value"
                exit 1
            fi
            ;;
        -t|--evaluate-data)
            if [[ -n "$2" && "$2" != -* ]]; then
                EVALUATE_DATA="$2"
                shift 2
            else
                echo "err: -t | --evaluate-data need a value"
                exit 1
            fi
            ;;
        -s|--score-on)
            if [[ -n "$2" && "$2" != -* ]]; then
                SCORE_ON="$2"
                shift 2
            else
                echo "err: -s | --score-on need a value"
                exit 1
            fi
            ;;
        *)
        echo "未知参数: $1"
        exit 1
        ;;
    esac
done

missing_args=()
[[ -z "$CONFIG_NAME" ]] && missing_args+=("--config")
[[ -z "$MODEL_PATH" ]] && missing_args+=("--model-path")
[[ -z "$EVALUATE_DATA" ]] && missing_args+=("--evaluate-data")
[[ -z "$SCORE_ON" ]] && missing_args+=("--score-on")

if [ ${#missing_args[@]} -ne 0 ]; then
    echo "Error: Missing required arguments: ${missing_args[*]}"
    exit 1
fi

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
if [[ ! -d "$PROJECT_BASE_PATH" ]]; then
    echo "Error: Project base path does not exist: $PROJECT_BASE_PATH"
    exit 1
fi

SCRIPT_PATH=$PROJECT_BASE_PATH/src
EPOCHS=3


run_evaluate() {
    local m_path="$1"
    local score_on="$2"
    local save_dir="$m_path/evaluation/$score_on/0_shots/$EVALUATE_DATA"

    echo ">>> [Evaluating] Model: $m_path"
    uv run python "$SCRIPT_PATH/evaluate.py" \
        --model "$m_path" \
        --mode full \
        --tokenizer "$CONFIG_NAME" \
        --test-data "$PROJECT_BASE_PATH/input/evaluate_data/unformated/$EVALUATE_DATA/test.json" \
        --score-on "$score_on" \
        --sample-num 1000 \
        -o "$save_dir"
}

echo ">>>>>> model config: $CONFIG_NAME"
echo ">>>>>> model path: $MODEL_PATH"
echo ">>>>>> evaluation data: $EVALUATE_DATA"

# w/o extended training
echo
echo
echo ">>> evaluating "
run_evaluate "$MODEL_PATH" "$SCORE_ON"

for sft_split in train test
do
    echo
    echo
    SFT_MODEL_PATH="$MODEL_PATH-$EVALUATE_DATA/sft_${sft_split}_ep${EPOCHS}"
    echo ">>> evaluating "
    run_evaluate "$SFT_MODEL_PATH" "$SCORE_ON"
done

# with extended training
for ext_train_split in train test
do
    echo
    echo
    EXT_MODEL_PATH="$MODEL_PATH-$EVALUATE_DATA/ext_${ext_train_split}_ep${EPOCHS}"
    echo ">>> evaluating "
    run_evaluate "$EXT_MODEL_PATH" "$SCORE_ON"

    for sft_split in train test
    do
        echo
        echo
        EXT_SFT_MODEL_PATH="$EXT_MODEL_PATH-sft_${sft_split}_ep${EPOCHS}"
        echo ">>> evaluating "
        run_evaluate "$EXT_SFT_MODEL_PATH" "$SCORE_ON"
    done
done


end_time=$(date +"%s")
echo "end time: $(date -d @"$end_time" +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"