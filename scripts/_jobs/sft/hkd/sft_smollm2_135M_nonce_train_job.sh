#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=50:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/sml_sft_nonce.log


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/src/train

# export WANDB_MODE=offline
MODEL_CONFIG="HuggingFaceTB/SmolLM2-135M"
MODEL_NAME=SmolLM2-135M-20B-nonce-bs1024


for SFT_DATA in based_suqad suqad_v2 race; do
    echo ">>> SFT on $SFT_DATA"
    uv run python train.py --config-name sft_train \
        base.path=$PROJECT_BASE_PATH \
        model.config="$MODEL_CONFIG" \
        model.init_model="$PROJECT_BASE_PATH/output/$MODEL_CONFIG/$MODEL_NAME" \
        data.name=$SFT_DATA
done


cd $PROJECT_BASE_PATH/scripts/eval
for SFT_DATA in based_suqad suqad_v2 race; do
    MODEL_PATH=$PROJECT_BASE_PATH/output/$MODEL_CONFIG/$MODEL_NAME-sft_${SFT_DATA}_train
    sh llm_eval.sh $MODEL_PATH
done