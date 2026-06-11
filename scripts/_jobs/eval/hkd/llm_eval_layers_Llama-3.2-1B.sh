#!/bin/bash
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30897
#PBS -j oe
#PBS -o logs/Llama-3.2-1B.log


source $HOME/.zshrc
cd $PROJECT_BASE_PATH/scripts/eval

MODEL_NAME="meta-llama/Llama-3.2-1B"

L1=1
L2=16
STEP=1
for i in $(seq $L1 $STEP $L2); do
    echo "keep $i layers"
    uv run python $PROJECT_BASE_PATH/src/train/drop_layers.py --config-name=default model.config=$MODEL_NAME model.keep_n_layers=$i base.path=$PROJECT_BASE_PATH
done

sh llm_eval_layers_qa.sh $MODEL_NAME $L1 $L2 $STEP
sh llm_eval_layers_blimp.sh $MODEL_NAME $L1 $L2 $STEP
