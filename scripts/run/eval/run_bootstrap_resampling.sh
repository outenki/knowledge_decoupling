#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/scripts/run/eval

# linguistic
# MODEL_NAME="random/rnd"
MODEL_NAME=$1
for EVAL_DATA in \
    verb_agreement \
    fce \
    fce_3gram \
    fce_5gram
do
    echo "Evaluating on $EVAL_DATA..."
    model_path="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME"
    sh "$SCRIPT_PATH/bootstrap_resampling.sh" \
        --config gpt2 \
        --score-on options \
        --model-path "$model_path" \
        --evaluate-data "$EVAL_DATA"
done


# qa
# ext_test-sft_test
for EVAL_DATA in \
    arc_easy \
    arc_challenge \
    qasc \
    commonsense_qa
do
    MODEL_NAME=$MODEL_NAME-$EVAL_DATA/ext_test-sft_test
    echo "Evaluating on $EVAL_DATA..."
    model_path="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME"
    sh "$SCRIPT_PATH/bootstrap_resampling.sh" \
        --config gpt2 \
        --score-on generation \
        --model-path "$model_path" \
        --evaluate-data "$EVAL_DATA"
done

# rag
# sft_train
for EVAL_DATA in \
    google_re_long \
    google_re_short
do
    MODEL_NAME=$MODEL_NAME-$EVAL_DATA/sft_test
    echo "Evaluating on $EVAL_DATA..."
    model_path="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME"
    sh "$SCRIPT_PATH/bootstrap_resampling.sh" \
        --config gpt2 \
        --score-on generation \
        --model-path "$model_path" \
        --evaluate-data "$EVAL_DATA"
done