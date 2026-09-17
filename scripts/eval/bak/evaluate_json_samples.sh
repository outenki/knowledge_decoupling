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
    m_path="$1"
    json_path="$m_path/evaluation/generation/0_shots/$EVALUATE_DATA/$DATA_FORMAT/evaluated_samples.json"
    echo "Evaluating samples in: $json_path"
    uv run python "$PROJECT_BASE_PATH/scripts/tools/evaluate_json_samples.py" $json_path
}

# w/o extended training
run_evaluate "$MODEL_PATH"

for sft_split in test train
do
    SFT_MODEL_PATH="$MODEL_PATH-$EVALUATE_DATA-$OUTPUT_SUFFIX/sft_${sft_split}_ep${EPOCHS}_lr${LEARNING_RATE}"
    run_evaluate "$SFT_MODEL_PATH"
done

# with extended training
for ext_train_split in test train
do
    EXT_MODEL_PATH="$MODEL_PATH-$EVALUATE_DATA-$OUTPUT_SUFFIX/ext_${ext_train_split}_ep${EPOCHS}_lr${LEARNING_RATE}"
    run_evaluate "$EXT_MODEL_PATH"

    for sft_split in test train
    do
        EXT_SFT_MODEL_PATH="$EXT_MODEL_PATH-sft_${sft_split}_ep${EPOCHS}_lr${LEARNING_RATE}"
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
