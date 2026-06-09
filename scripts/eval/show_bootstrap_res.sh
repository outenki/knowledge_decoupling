#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/scripts/run/eval

# linguistic
# MODEL_NAME="random/rnd"
MODEL_NAME=$1

echo "verb agreement fce fce_3gram fce_5gram"
EVAL_DATA="verb_agreement"
EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME/evaluation/options/0_shots/$EVAL_DATA"
uv run python $SCRIPT_PATH/show_bootstrap_res.py $EVAL_PATH "accuracy"

EVAL_DATA="fce"
EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME/evaluation/options/0_shots/$EVAL_DATA"
uv run python $SCRIPT_PATH/show_bootstrap_res.py $EVAL_PATH "accuracy"

EVAL_DATA="fce_3gram"
EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME/evaluation/options/0_shots/$EVAL_DATA"
uv run python $SCRIPT_PATH/show_bootstrap_res.py $EVAL_PATH "accuracy"

EVAL_DATA="fce_5gram"
EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME/evaluation/options/0_shots/$EVAL_DATA"
uv run python $SCRIPT_PATH/show_bootstrap_res.py $EVAL_PATH "accuracy"


# qa
# ext_test-sft_test
echo "(w/o) arc_easy arc_challenge qasc commonsense_qa"
for EVAL_DATA in \
    arc_easy \
    arc_challenge \
    qasc \
    commonsense_qa
do
    EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME-$EVAL_DATA/sft_test_ep3/evaluation/generation/0_shots/$EVAL_DATA"
    uv run python $SCRIPT_PATH/show_bootstrap_res.py $EVAL_PATH "f1"
done

echo "(w/o) arc_easy arc_challenge qasc commonsense_qa"
for EVAL_DATA in \
    arc_easy \
    arc_challenge \
    qasc \
    commonsense_qa
do
    EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME-$EVAL_DATA/ext_test_ep3-sft_test_ep3/evaluation/generation/0_shots/$EVAL_DATA"
    uv run python $SCRIPT_PATH/show_bootstrap_res.py $EVAL_PATH "f1"
done

# rag
# sft_train
echo "google_re_long_context google_re_short_context"
for EVAL_DATA in \
    google_re_long_context \
    google_re_short_context
do
    EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME-$EVAL_DATA/sft_train_ep3/evaluation/generation/0_shots/$EVAL_DATA"
    uv run python $SCRIPT_PATH/show_bootstrap_res.py $EVAL_PATH "f1"
done
