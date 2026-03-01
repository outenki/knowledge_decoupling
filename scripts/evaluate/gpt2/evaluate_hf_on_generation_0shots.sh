#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"

MODEL_NAME=gpt2
SCORE_ON=generation
FEWSHOTS=0
SAMPLE_NUM=1000
MODE="full"
SUFFIX="_$MODE"


for eval_name in \
    squad_v2_ctxt_answerable \
    cwq \
    qa_arc_easy \
    qa_arc_challenge \
    metaqa_1hop \
    metaqa_2hop \
    metaqa_3hop \
    mintaka_multihop \
    qa_qasc
do
    echo
    echo "====== Evaluating $model_folder of $MODEL_NAME ======"
    model_path="gpt2"
    uv run python "$PROJECT_BASE_PATH"/src/evaluate.py \
        --model "$model_path" \
        --mode $MODE \
        --tokenizer $MODEL_NAME \
        --test-data "$PROJECT_BASE_PATH"/input/evaluate_data/unformated/$eval_name/test.json \
        --score-on $SCORE_ON \
        --sample-num $SAMPLE_NUM \
        -o "$PROJECT_BASE_PATH"/output/$MODEL_NAME/HuggingFace/hf/evaluation$SUFFIX/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    echo "============ $eval_name ============"
    for model_folder in \
        hf-sft_qa_context_test \
        hf-sft_qa_wo_context_test \
        hf-sft_qa_wo_context_train \
        hf-sft_qa_context_train \
        hf-ext_qa_test_ep3 \
        hf-ext_qa_test_ep3-sft_qa_wo_context_train \
        hf-ext_qa_test_ep3-sft_qa_wo_context_test \
        hf-ext_qa_test_ep3-sft_qa_w_context_train \
        hf-ext_qa_test_ep3-sft_qa_w_context_test
    do
        echo "====== Evaluating $model_folder of $MODEL_NAME ======"
        model_path=$PROJECT_BASE_PATH/output/$MODEL_NAME/HuggingFace/$model_folder
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
