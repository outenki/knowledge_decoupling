#!/bin/bash

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--evaluate-data)
            if [[ -n "$2" && "$2" != -* ]]; then
                EVALUATE_DATA="$2"
                shift 2
            else
                echo "err: -d | --evaluate-data need a value"
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

if [ ${#missing_args[@]} -ne 0 ]; then
    echo "Error: Missing required arguments: ${missing_args[*]}"
    exit 1
fi

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
if [[ ! -d "$PROJECT_BASE_PATH" ]]; then
    echo "Error: Project base path does not exist: $PROJECT_BASE_PATH"
    exit 1
fi

EPOCHS=3


print_evaluate() {
    local m_path="$1"
    local save_dir="$m_path/evaluation/$EVALUATE_ON/0_shots/$EVALUATE_DATA"
    f1=$(cat "$save_dir/evaluation_summary.json"|grep f1|cut -d":" -f 2|cut -d"," -f1)
    acc=$(cat "$save_dir/evaluation_summary.json"|grep accuracy|cut -d":" -f 2|cut -d"," -f1)
    if [ "$EVALUATE_ON" == "generation" ]; then
        echo "F1 Score for $m_path: $f1"
    elif [ "$EVALUATE_ON" == "options" ]; then
        echo "Accuracy for $m_path: $acc"
    fi
}

CONFIG_PATH="$PROJECT_BASE_PATH/output/gpt2"
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
        SFT_MODEL_PATH="$MODEL_PATH-$EVALUATE_DATA/sft_${sft_split}_ep${EPOCHS}"
        print_evaluate "$SFT_MODEL_PATH"
    done

    # with extended training
    for ext_train_split in test train
    do
        EXT_MODEL_PATH="$MODEL_PATH-$EVALUATE_DATA/ext_${ext_train_split}_ep${EPOCHS}"
        print_evaluate "$EXT_MODEL_PATH"

        for sft_split in test train
        do
            EXT_SFT_MODEL_PATH="$EXT_MODEL_PATH-sft_${sft_split}_ep${EPOCHS}"
            print_evaluate "$EXT_SFT_MODEL_PATH"
        done
    done
done