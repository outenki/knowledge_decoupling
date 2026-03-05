#!/bin/bash
echo "====== training on qa_w_context_train ======"
start_time=$(date +"%s")
echo "start time: $(date -d @"$start_time" +"%D %T")"

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            if [[ -n "$2" && "$2" != -* ]]; then
                CONFIG_NAME="$2"
                shift 2
            else
                echo "err: --model need a value"
                exit 1
            fi
            ;;
        -p|--model-path)
            if [[ -n "$2" && "$2" != -* ]]; then
                MODEL_PATH="$2"
                shift 2
            else
                echo "err: --model-path need a value"
                exit 1
            fi
            ;;
        -e|--ext-train-data)
            if [[ -n "$2" && "$2" != -* ]]; then
                EXT_TRAIN_DATA="$2"
                shift 2
            else
                echo "err: --ext-train-data need a value"
                exit 1
            fi
            ;;
        -s|--sft-data)
            if [[ -n "$2" && "$2" != -* ]]; then
                SFT_DATA="$2"
                shift 2
            else
                echo "err: --sft-data need a value"
                exit 1
            fi
            ;;
        -t|--evaluate-data)
            if [[ -n "$2" && "$2" != -* ]]; then
                EVALUATE_DATA="$2"
                shift 2
            else
                echo "err: --evaluate-data need a value"
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
[[ -z "$CONFIG_NAME" ]] && missing_args+=("--model")
[[ -z "$MODEL_PATH" ]] && missing_args+=("--model-path")
[[ -z "$EVALUATE_DATA" ]] && missing_args+=("--evaluate-data")
[[ -z "$SFT_DATA" ]] && missing_args+=("--sft-data")
[[ -z "$EXT_TRAIN_DATA" ]] && missing_args+=("--ext-train-data")

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
    local save_dir="$m_path/evaluation/generation/0_shots/$EVALUATE_DATA"
    if [[ ! -f "$m_path" ]]; then
        echo "Error: model file not found at: $m_path"
        echo "Aborting current evaluating task."
        return 1
    fi
    
    echo ">>> [Evaluating] Model: $m_path"
    uv run python "$PROJECT_BASE_PATH/src/evaluate.py" \
        --model "$m_path" \
        --mode full \
        --tokenizer "$CONFIG_NAME" \
        --test-data "$PROJECT_BASE_PATH/input/evaluate_data/unformated/$EVALUATE_DATA/test.json" \
        --score-on generation \
        --sample-num 1000 \
        -o "$save_dir"
}

run_train() {
    local in_model="$1"
    local data_path="$2"
    local out_model="$3"

    if [[ ! -f "$data_path" ]]; then
        echo "Error: Training data file not found at: $data_path"
        echo "Aborting current training task."
        return 1
    fi

    echo ">>> [Training] Input: $in_model -> Output: $out_model"
    uv run python "$SCRIPT_PATH/train.py" \
        --speedup \
        -pad \
        -cn "$CONFIG_NAME" \
        -im "$in_model" \
        -dp "$data_path" \
        -dl 0 \
        -e "$EPOCHS" \
        -o "$out_model"

    if [ $? -ne 0 ]; then
        echo "err: training failed for $in_model with data $data_path"
        exit 1
    fi
}

echo ">>>>>> model config: $CONFIG_NAME"
echo ">>>>>> model path: $MODEL_PATH"
echo ">>>>>> extended training data: $EXT_TRAIN_DATA"
echo ">>>>>> SFT data: $SFT_DATA"
echo ">>>>>> evaluation data: $EVALUATE_DATA"
echo ">>>>>> option: output directory: $OUTPUT_DIR"

# w/o extended training
echo ">>>>>> model config: $CONFIG_NAME"
echo ">>>>>> model path: $MODEL_PATH"
echo ">>> evaluating "
run_evaluate "$MODEL_PATH"
for sft_split in train test
do
    echo ">>>>>> model config: $CONFIG_NAME"
    echo ">>>>>> model path: $MODEL_PATH"
    echo ">>>>>> SFT data: $SFT_DATA/$sft_split.json"
    echo ">>> SFT training"
    SFT_MODEL_PATH="$MODEL_PATH-$EVALUATE_DATA-sft_${sft_split}_ep${EPOCHS}"
    run_train "$MODEL_PATH" "$SFT_DATA/$sft_split.json" "$SFT_MODEL_PATH"
    echo ">>> evaluating "
    run_evaluate "$SFT_MODEL_PATH"
done

# with extended training
for ext_train_split in train test
do
    echo ">>>>>> model config: $CONFIG_NAME"
    echo ">>>>>> model path: $MODEL_PATH"
    echo ">>>>>> extended training data: $EXT_TRAIN_DATA/$ext_train_split.json"
    echo ">>> extended training"
    EXT_MODEL_PATH="$MODEL_PATH-$EVALUATE_DATA-ext_${ext_train_split}_ep${EPOCHS}"
    run_train "$MODEL_PATH" "$EXT_TRAIN_DATA/$ext_train_split.json" "$EXT_MODEL_PATH"
    echo ">>> evaluating "
    run_evaluate "$EXT_MODEL_PATH"

    for sft_split in train test
    do
        echo ">>>>>> model config: $CONFIG_NAME"
        echo ">>>>>> model path: $MODEL_PATH"
        echo ">>>>>> SFT data: $SFT_DATA/$sft_split.json"
        echo ">>> SFT training"
        EXT_SFT_MODEL_PATH="$EXT_MODEL_PATH-sft_${sft_split}_ep${EPOCHS}"
        run_train "$EXT_MODEL_PATH" "$SFT_DATA/$sft_split.json" "$EXT_SFT_MODEL_PATH"
        echo ">>> evaluating "
        run_evaluate "$EXT_SFT_MODEL_PATH"
    done
done


end_time=$(date +"%s")
echo "end time: $(date -d @"$end_time" +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"