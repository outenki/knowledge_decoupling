MODEL_PATH=$1
START_LAYER=$2
END_LAYER=$3
STEP=$4

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd $MODEL_PATH
for i in $(seq $START_LAYER $STEP $END_LAYER); do
    echo
    echo "===eval layers $i===";
    uv run accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained=layers_$i/base \
        --tasks arc_easy,arc_challenge,commonsense_qa,boolq,race,drop \
        --output_path layers_$i/base/eval/qa
done