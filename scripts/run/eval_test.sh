#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"

MODEL_NAME=gpt2
FEWSHOTS=0
SAMPLE_NUM=1000
MODE="full"
SUFFIX="_temp"


for SCORE_ON in generation options
do
    echo
    for model_folder in \
        hf-sft_qa_context_train \
        hf-ext_qa_test_ep3-sft_qa_w_context_train
    do
        echo "====== Evaluating $model_folder of $MODEL_NAME ======"
        model_path=$PROJECT_BASE_PATH/output/$MODEL_NAME/ss/smolLM2_135M_sents_shuffled_bs1024_ep1-commonsense_qa/ext_test_ep3-sft_test_ep3
        uv run python "$PROJECT_BASE_PATH"/src/evaluate.py \
            --model "$model_path" \
            --mode $MODE \
            --tokenizer $MODEL_NAME \
            --test-data "$PROJECT_BASE_PATH"/input/evaluate_data/unformated/$eval_name/test.json \
            --score-on $SCORE_ON \
            --sample-num $SAMPLE_NUM \
            -o "$model_path"/evaluation$SUFFIX/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    done
done
