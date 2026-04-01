#!/bin/bash
start_time=$(date +"%s")
echo "start time: $(date -d @"$start_time" +"%D %T")"

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/scripts/run/eval

# linguistic
# MODEL_NAME="random/rnd"
MODEL_NAME=$1

EVAL_DATA="verb_agreement"
echo "Bootstrap result on $EVAL_DATA"
EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME/evaluation/options/0_shots/$EVAL_DATA"
uv run python $SCRIPT_PATH/show_bootstrap_res.py $EVAL_PATH "accuracy"

EVAL_DATA="fce"
EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME/evaluation/options/0_shots/$EVAL_DATA"
echo "Bootstrap result on $EVAL_DATA"
uv run python $SCRIPT_PATH/show_bootstrap_res.py $EVAL_PATH "accuracy"

EVAL_DATA="fce_3gram"
EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME/evaluation/options/0_shots/$EVAL_DATA"
echo "Bootstrap result on $EVAL_DATA"
uv run python $SCRIPT_PATH/show_bootstrap_res.py $EVAL_PATH "accuracy"

EVAL_DATA="fce_5gram"
EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME/evaluation/options/0_shots/$EVAL_DATA"
echo "Bootstrap result on $EVAL_DATA"
uv run python $SCRIPT_PATH/show_bootstrap_res.py $EVAL_PATH "accuracy"


# qa
# ext_test-sft_test
for EVAL_DATA in \
    arc_easy \
    arc_challenge \
    qasc \
    commonsense_qa
do
    echo "Bootstrap result on $EVAL_DATA"
    EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME-$EVAL_DATA/ext_test_ep3-sft_test_ep3/evaluation/generation/0_shots/$EVAL_DATA"
    uv run python $SCRIPT_PATH/show_bootstrap_res.py $EVAL_PATH "f1"
done

# rag
# sft_train
for EVAL_DATA in \
    google_re_long_context \
    google_re_short_context
do
    echo "Bootstrap result on $EVAL_DATA"
    EVAL_PATH="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME-$EVAL_DATA/sft_test_ep3/evaluation/generation/0_shots/$EVAL_DATA"
    uv run python $SCRIPT_PATH/show_bootstrap_res.py $EVAL_PATH "f1"
done

end_time=$(date +"%s")
echo "end time: $(date -d @"$end_time" +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"