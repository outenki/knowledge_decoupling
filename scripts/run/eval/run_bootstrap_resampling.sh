#!/bin/bash
start_time=$(date +"%s")
echo "start time: $(date -d @"$start_time" +"%D %T")"

PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH="$PROJECT_BASE_PATH"/scripts/run/eval

# linguistic
# MODEL_NAME="random/rnd"
MODEL_NAME=$1

EVAL_DATA="verb_agreement"
SAMPLE_NUM=200
echo "Evaluating on $EVAL_DATA..."
model_path="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME"
sh "$SCRIPT_PATH/bootstrap_resampling.sh" \
    --config gpt2 \
    --score-on options \
    --model-path "$model_path" \
    --sample-num "$SAMPLE_NUM" \
    --evaluate-data "$EVAL_DATA"

EVAL_DATA="fce"
SAMPLE_NUM=150
echo "Evaluating on $EVAL_DATA..."
model_path="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME"
sh "$SCRIPT_PATH/bootstrap_resampling.sh" \
    --config gpt2 \
    --score-on options \
    --model-path "$model_path" \
    --sample-num "$SAMPLE_NUM" \
    --evaluate-data "$EVAL_DATA"

EVAL_DATA="fce_3gram"
SAMPLE_NUM=100
echo "Evaluating on $EVAL_DATA..."
model_path="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME"
sh "$SCRIPT_PATH/bootstrap_resampling.sh" \
    --config gpt2 \
    --score-on options \
    --model-path "$model_path" \
    --sample-num "$SAMPLE_NUM" \
    --evaluate-data "$EVAL_DATA"

EVAL_DATA="fce_5gram"
SAMPLE_NUM=80
echo "Evaluating on $EVAL_DATA..."
model_path="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME"
sh "$SCRIPT_PATH/bootstrap_resampling.sh" \
    --config gpt2 \
    --score-on options \
    --model-path "$model_path" \
    --sample-num "$SAMPLE_NUM" \
    --evaluate-data "$EVAL_DATA"


SAMPLE_NUM=400
# qa
# ext_test-sft_test
for EVAL_DATA in \
    arc_easy \
    arc_challenge \
    qasc \
    commonsense_qa
do
    MODEL_NAME_EXT_SFT=$MODEL_NAME-$EVAL_DATA/ext_test_ep3-sft_test_ep3
    echo "Evaluating on $EVAL_DATA..."
    model_path="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME_EXT_SFT"
    sh "$SCRIPT_PATH/bootstrap_resampling.sh" \
        --config gpt2 \
        --score-on generation \
        --model-path "$model_path" \
        --sample-num "$SAMPLE_NUM" \
        --evaluate-data "$EVAL_DATA"
done

# rag
# sft_train
for EVAL_DATA in \
    google_re_long_context \
    google_re_short_context
do
    MODEL_NAME_SFT=$MODEL_NAME-$EVAL_DATA/sft_train_ep3
    echo "Evaluating on $EVAL_DATA..."
    model_path="$PROJECT_BASE_PATH/output/gpt2/$MODEL_NAME_SFT"
    sh "$SCRIPT_PATH/bootstrap_resampling.sh" \
        --config gpt2 \
        --score-on generation \
        --model-path "$model_path" \
        --sample-num "$SAMPLE_NUM" \
        --evaluate-data "$EVAL_DATA"
done

end_time=$(date +"%s")
echo "end time: $(date -d @"$end_time" +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"