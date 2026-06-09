MODEL_NAME=$1
TASKS=$2
START_LAYER=$3
END_LAYER=$4
STEP=$5

cd $PROJECT_BASE_PATH/output/$MODEL_NAME
for i in $(seq $START_LAYER $STEP $END_LAYER); do
    echo
    echo "===eval layers $i===";
    uv run accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained=layers_$i/base \
        --tasks $TASKS \
        --output_path layers_$i/base/eval/blimp
done