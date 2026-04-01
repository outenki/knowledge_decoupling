#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/src/tools

# linguistic
# MODEL_NAME="random/rnd"
MODEL_NAME=$1

HF_MODEL="HuggingFace/hf"
SML_MODEL="smolLM2/smolLM2_bs1024_dl0_ep1"

for EVAL_DATA in \
    verb_agreement \
    fce \
    fce_3gram \
    fce_5gram
do
    echo "Running t-test on $EVAL_DATA"
    EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME/evaluation/options/0_shots/$EVAL_DATA"
    HF_PATH="$PROJECT_BASE_PATH/output/gpt2/$HF_MODEL/evaluation/options/0_shots/$EVAL_DATA"
    SML_PATH="$PROJECT_BASE_PATH/output/gpt2/$SML_MODEL/evaluation/options/0_shots/$EVAL_DATA"
    uv run python "$SCRIPT_PATH"/t_test.py "accuracy" "$EVAL_PATH" "$HF_PATH" "$EVAL_PATH/t_test/hf"
    uv run python "$SCRIPT_PATH"/t_test.py "accuracy" "$EVAL_PATH" "$SML_PATH" "$EVAL_PATH/t_test/sml"
done


# qa
# ext_test-sft_test
for EVAL_DATA in \
    arc_easy \
    arc_challenge \
    qasc \
    commonsense_qa
do
    echo "Running t-test on $EVAL_DATA (w/o)"
    EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME-$EVAL_DATA/sft_test_ep3/evaluation/generation/0_shots/$EVAL_DATA"
    HF_PATH="$PROJECT_BASE_PATH/output/gpt2/$HF_MODEL-$EVAL_DATA/sft_test_ep3/evaluation/generation/0_shots/$EVAL_DATA"
    SML_PATH="$PROJECT_BASE_PATH/output/gpt2/$SML_MODEL-$EVAL_DATA/sft_test_ep3/evaluation/generation/0_shots/$EVAL_DATA"
    uv run python "$SCRIPT_PATH"/t_test.py "accuracy" "$EVAL_PATH" "$HF_PATH" "$EVAL_PATH/t_test/hf"
    uv run python "$SCRIPT_PATH"/t_test.py "accuracy" "$EVAL_PATH" "$SML_PATH" "$EVAL_PATH/t_test/sml"
done

echo "(w/o) arc_easy arc_challenge qasc commonsense_qa"
for EVAL_DATA in \
    arc_easy \
    arc_challenge \
    qasc \
    commonsense_qa
do
    echo "Running t-test on $EVAL_DATA (w/)"
    EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME-$EVAL_DATA/ext_test_ep3-sft_test_ep3/evaluation/generation/0_shots/$EVAL_DATA"
    HF_PATH="$PROJECT_BASE_PATH/output/gpt2/$HF_MODEL-$EVAL_DATA/ext_test_ep3-sft_test_ep3/evaluation/generation/0_shots/$EVAL_DATA"
    SML_PATH="$PROJECT_BASE_PATH/output/gpt2/$SML_MODEL-$EVAL_DATA/ext_test_ep3-sft_test_ep3/evaluation/generation/0_shots/$EVAL_DATA"
    uv run python "$SCRIPT_PATH"/t_test.py "accuracy" "$EVAL_PATH" "$HF_PATH" "$EVAL_PATH/t_test/hf"
    uv run python "$SCRIPT_PATH"/t_test.py "accuracy" "$EVAL_PATH" "$SML_PATH" "$EVAL_PATH/t_test/sml"
done

# rag
# sft_train
echo "google_re_long_context google_re_short_context"
for EVAL_DATA in \
    google_re_long_context \
    google_re_short_context
do
    echo "Running t-test on $EVAL_DATA (w/)"
    EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME-$EVAL_DATA/sft_train_ep3/evaluation/generation/0_shots/$EVAL_DATA"
    HF_PATH="$PROJECT_BASE_PATH/output/gpt2/$HF_MODEL-$EVAL_DATA/sft_train_ep3/evaluation/generation/0_shots/$EVAL_DATA"
    SML_PATH="$PROJECT_BASE_PATH/output/gpt2/$SML_MODEL-$EVAL_DATA/sft_train_ep3/evaluation/generation/0_shots/$EVAL_DATA"
    uv run python "$SCRIPT_PATH"/t_test.py "accuracy" "$EVAL_PATH" "$HF_PATH" "$EVAL_PATH/t_test/hf"
    uv run python "$SCRIPT_PATH"/t_test.py "accuracy" "$EVAL_PATH" "$SML_PATH" "$EVAL_PATH/t_test/sml"
done
