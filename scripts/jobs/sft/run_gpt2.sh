#!/bin/bash
start_time=$(date +"%s")
echo "start time: $(date -d @"$start_time" +"%D %T")"
export WANDB_MODE=offline

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
        -e|--ext-train-data)
            if [[ -n "$2" && "$2" != -* ]]; then
                EXT_TRAIN_DATA="$2"
                shift 2
            else
                echo "err: -e | --ext-train-data need a value"
                exit 1
            fi
            ;;
        -s|--sft-data)
            if [[ -n "$2" && "$2" != -* ]]; then
                SFT_DATA="$2"
                shift 2
            else
                echo "err: -s | --sft-data need a value"
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
        -f|--evaluate-data-format)
            if [[ -n "$2" && "$2" != -* ]]; then
                DATA_FORMAT="$2"
                shift 2
            else
                echo "err: -f | --evaluate-data-format need a value"
                exit 1
            fi
            ;;
        -l|--learning-rate)
            if [[ -n "$2" && "$2" != -* ]]; then
                LEARNING_RATE="$2"
                shift 2
            else
                echo "err: -l | --learning-rate need a value"
                exit 1
            fi
            ;;
        -o|--output-suffix)
            if [[ -n "$2" && "$2" != -* ]]; then
                OUTPUT_SUFFIX="$2"
                shift 2
            else
                echo "err: -o | --output-suffix need a value"
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
[[ -z "$LEARNING_RATE" ]] && missing_args+=("--learning-rate")
[[ -z "$EVALUATE_DATA" ]] && missing_args+=("--evaluate-data")
[[ -z "$SFT_DATA" ]] && missing_args+=("--sft-data")
[[ -z "$EXT_TRAIN_DATA" ]] && missing_args+=("--ext-train-data")
[[ -z "$OUTPUT_SUFFIX" ]] && missing_args+=("--output-suffix")
[[ -z "$DATA_FORMAT" ]] && missing_args+=("--evaluate-data-format")

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
    local save_dir="$m_path/evaluation/options/0_shots/$EVALUATE_DATA/$DATA_FORMAT"
 
    echo ">>> [Evaluating on options] Model: $m_path"
    uv run python "$SCRIPT_PATH/eval/evaluate.py" \
        --model "$m_path" \
        --mode full \
        --tokenizer "$CONFIG_NAME" \
        --test-data "$PROJECT_BASE_PATH/input/evaluate_data/unformated/$EVALUATE_DATA/test.json" \
        --score-on options \
        --sample-num 1000 \
        --data-format "$DATA_FORMAT" \
        -o "$save_dir"
    echo ">>> [Evaluating on generation] Model: $m_path"
    local save_dir="$m_path/evaluation/generation/0_shots/$EVALUATE_DATA/$DATA_FORMAT"
    uv run python "$SCRIPT_PATH/eval/evaluate.py" \
        --model "$m_path" \
        --mode full \
        --tokenizer "$CONFIG_NAME" \
        --test-data "$PROJECT_BASE_PATH/input/evaluate_data/unformated/$EVALUATE_DATA/test.json" \
        --score-on generation \
        --sample-num 1000 \
        --data-format "$DATA_FORMAT" \
        -o "$save_dir"
}

run_train() {
    local in_model="$1"
    local data_path="$2"
    local out_model="$3"
    local learning_rate="$4"

    echo ">>> [Training] Input: $in_model -> Output: $out_model"
    uv run python "$SCRIPT_PATH/train/train.py" \
        -cn "$CONFIG_NAME" \
        -im "$in_model" \
        -dp "$data_path" \
        -dl 0 \
        -e "$EPOCHS" \
        -lr "$learning_rate" \
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

# w/o extended training
echo ">>>>>> model config: $CONFIG_NAME"
echo ">>>>>> model path: $MODEL_PATH"
echo ">>> evaluating "
run_evaluate "$MODEL_PATH"

for sft_split in test train
do
    echo
    echo
    echo ">>>>>> model config: $CONFIG_NAME"
    echo ">>>>>> model path: $MODEL_PATH"
    echo ">>>>>> SFT data: $SFT_DATA/$sft_split"
    echo ">>> SFT training"
    SFT_MODEL_PATH="$MODEL_PATH-$EVALUATE_DATA-$OUTPUT_SUFFIX/sft_${sft_split}_ep${EPOCHS}_lr${LEARNING_RATE}"
    run_train "$MODEL_PATH" "$SFT_DATA/$sft_split" "$SFT_MODEL_PATH" $LEARNING_RATE
    echo ">>> evaluating "
    run_evaluate "$SFT_MODEL_PATH"
done

# with extended training
for ext_train_split in test train
do
    echo
    echo
    echo ">>>>>> model config: $CONFIG_NAME"
    echo ">>>>>> model path: $MODEL_PATH"
    echo ">>>>>> extended training data: $EXT_TRAIN_DATA/$ext_train_split"
    echo ">>> extended training"
    EXT_MODEL_PATH="$MODEL_PATH-$EVALUATE_DATA-$OUTPUT_SUFFIX/ext_${ext_train_split}_ep${EPOCHS}_lr${LEARNING_RATE}"
    run_train "$MODEL_PATH" "$EXT_TRAIN_DATA/$ext_train_split" "$EXT_MODEL_PATH" $LEARNING_RATE
    echo ">>> evaluating "
    run_evaluate "$EXT_MODEL_PATH"

    for sft_split in test train
    do
        echo
        echo
        echo ">>>>>> model config: $CONFIG_NAME"
        echo ">>>>>> model path: $MODEL_PATH"
        echo ">>>>>> SFT data: $SFT_DATA/$sft_split"
        echo ">>> SFT training"
        EXT_SFT_MODEL_PATH="$EXT_MODEL_PATH-sft_${sft_split}_ep${EPOCHS}_lr${LEARNING_RATE}"
        run_train "$EXT_MODEL_PATH" "$SFT_DATA/$sft_split" "$EXT_SFT_MODEL_PATH" $LEARNING_RATE
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
