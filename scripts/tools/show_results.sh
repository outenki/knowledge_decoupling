#!/bin/bash

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config-name)
            if [[ -n "$2" && "$2" != -* ]]; then
                CONFIG_NAME="$2"
                shift 2
            else
                echo "err: -c | --config-name need a value"
                exit 1
            fi
            ;;
        -d|--evaluate-data)
            if [[ -n "$2" && "$2" != -* ]]; then
                EVALUATE_DATA="$2"
                shift 2
            else
                echo "err: -d | --evaluate-data need a value"
                exit 1
            fi
            ;;
        -f|--evaluate-data-format)
            if [[ -n "$2" && "$2" != -* ]]; then
                EVALUATE_DATA_FORMAT="$2"
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
        -e|--epochs)
            if [[ -n "$2" && "$2" != -* ]]; then
                EPOCHS="$2"
                shift 2
            else
                echo "err: -e | --epochs need a value"
                exit 1
            fi
            ;;
        -on|--evaluate-on)
            if [[ -n "$2" && "$2" != -* ]]; then
                EVALUATE_ON="$2"
                shift 2
            else
                echo "err: -on | --evaluate-on need a value"
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
[[ -z "$EVALUATE_DATA" ]] && missing_args+=("--evaluate-data")
[[ -z "$EVALUATE_ON" ]] && missing_args+=("--evaluate-on")
[[ -z "$CONFIG_NAME" ]] && missing_args+=("--config-name")
[[ -z "$EVALUATE_DATA_FORMAT" ]] && missing_args+=("--evaluate-data-format")
[[ -z "$LEARNING_RATE" ]] && missing_args+=("--learning-rate")
[[ -z "$EPOCHS" ]] && missing_args+=("--epochs")

if [ ${#missing_args[@]} -ne 0 ]; then
    echo "Error: Missing required arguments: ${missing_args[*]}"
    exit 1
fi

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
if [[ ! -d "$PROJECT_BASE_PATH" ]]; then
    echo "Error: Project base path does not exist: $PROJECT_BASE_PATH"
    exit 1
fi


print_evaluate() {
    local m_path="$1"
    local save_dir="$m_path/evaluation/$EVALUATE_ON/0_shots/$EVALUATE_DATA/$EVALUATE_DATA_FORMAT"
    f1=$(cat "$save_dir/evaluation_summary.json"|grep f1|cut -d":" -f 2|cut -d"," -f1)
    acc=$(cat "$save_dir/evaluation_summary.json"|grep accuracy|cut -d":" -f 2|cut -d"," -f1)
    if [ "$EVALUATE_ON" == "generation" ]; then
        echo "F1 Score for $m_path: $f1"
    elif [ "$EVALUATE_ON" == "options" ]; then
        echo "Accuracy for $m_path: $acc"
    fi
}

CONFIG_PATH="$PROJECT_BASE_PATH/output/$CONFIG_NAME"
for MODEL_PATH in \
    $CONFIG_PATH/random/rnd \
    $CONFIG_PATH/HuggingFace/hf \
    $CONFIG_PATH/smolLM2/smolLM2_bs1024_dl0_ep1 \
    $CONFIG_PATH/ss/smolLM2_135M_sents_shuffled_bs1024_ep1 \
    $CONFIG_PATH/nonce/smolLM2_nonce_mn3_bs1024_dl0_ep1
do
    print_evaluate "$MODEL_PATH"
    for sft_split in test train
    do
        SFT_MODEL_PATH="$MODEL_PATH-$EVALUATE_DATA-$EVALUATE_DATA_FORMAT/sft_${sft_split}_ep${EPOCHS}_lr${LEARNING_RATE}"
        print_evaluate "$SFT_MODEL_PATH"
    done

    # with extended training
    for ext_train_split in test train
    do
        EXT_MODEL_PATH="$MODEL_PATH-$EVALUATE_DATA-$EVALUATE_DATA_FORMAT/ext_${ext_train_split}_ep${EPOCHS}_lr${LEARNING_RATE}"
        print_evaluate "$EXT_MODEL_PATH"

        for sft_split in test train
        do
            EXT_SFT_MODEL_PATH="$EXT_MODEL_PATH-sft_${sft_split}_ep${EPOCHS}_lr${LEARNING_RATE}"
            print_evaluate "$EXT_SFT_MODEL_PATH"
        done
    done
done
