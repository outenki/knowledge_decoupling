#!/bin/bash
start_time=$(date +"%s")
echo "start time: $(date -d @"$start_time" +"%D %T")"

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


run_evaluate() {
    local m_path="$1"
    local score_on="$2"
    local save_dir="$m_path/evaluation/$score_on/0_shots/$EVALUATE_DATA"

    echo ">>> [Evaluating] Model: $m_path"
    uv run python "$SCRIPT_PATH/evaluate.py" \
        --model "$m_path" \
        --mode full \
        --tokenizer "$CONFIG_NAME" \
        --test-data "$PROJECT_BASE_PATH/input/evaluate_data/linguistic/$EVALUATE_DATA/test.json" \
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
run_evaluate "$MODEL_PATH" options


end_time=$(date +"%s")
echo "end time: $(date -d @"$end_time" +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"