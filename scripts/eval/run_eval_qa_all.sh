#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/scripts/run/eval

# SCORE_ON="generation"
SCORE_ON="options"

CONFIG_NAME="Qwen/Qwen3.5-0.8B"

    # commonsense_qa
for EVAL_DATA in \
    qasc \
    arc_easy \
    arc_challenge
do
    echo ">>>>>>>>>>>>>>>>> Evaluating on $EVAL_DATA... <<<<<<<<<<<<<<<<<<"
        # random/rnd \
        # nonce/smolLM2_nonce_mn3_bs1024_dl0_ep1 \
        # ss/smolLM2_135M_sents_shuffled_bs1024_ep1 \
        # smolLM2/smolLM2_bs1024_dl0_ep1
    for model in \
        HuggingFace/hf
    do
        model_path="$PROJECT_BASE_PATH/output/$CONFIG_NAME/$model"
        sh "$SCRIPT_PATH/eval_qa.sh" \
            --config "$CONFIG_NAME" \
            --model-path "$model_path" \
            --evaluate-data "$EVAL_DATA" \
            --score-on "$SCORE_ON"
    done
done
