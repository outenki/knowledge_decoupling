#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-/home/pj25000107/ku50001566/projects/knowledge_decoupling}"

MODEL_NAME=gpt2
SCORE_ON=generation
FEWSHOTS=0
SAMPLE_NUM=1000
MODE="simple"
SUFFIX="_20260105"

# for eval_name in arc_easy arc_challenge qasc boolq squad_v2; do
for eval_name in qasc; do
    echo
    echo "============ $eval_name ============"

    # hf model
    for model_folder in \
        hf-sft-mix_qa_without_options-ep3 \
        random-sft-mix_qa_without_options-ep3 \
        nonce/smolLM2-nonce-mn3-bs1024-dl0-ep1-sft-mix_qa_without_options-ep3 \
        smolLM2/smolLM2-bs1024-dl0-ep1-sft-mix_qa_without_options-ep3
    do
        echo "====== Evaluating $model_folder of $MODEL_NAME ======"
        model_path=$PROJECT_BASE_PATH/output/$MODEL_NAME/$model_folder
        /home/pj25000107/ku50001566/.local/bin/uv run python $PROJECT_BASE_PATH/src/evaluate.py \
            --model $model_path \
            --mode $MODE \
            --tokenizer $MODEL_NAME \
            --test-data $PROJECT_BASE_PATH/input/evaluate_data/qa/without_options/$eval_name/test.json \
            --score-on $SCORE_ON \
            --sample-num $SAMPLE_NUM \
            -o $model_path/evaluation$SUFFIX/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    done

    # for model_folder in \
    #     random-sft-mix_qa_without_options-ep3 \
    #     smolLM2/smolLM2-bs1024-dl0-ep1-sft-mix_qa_without_options-ep3 \
    #     nonce/smolLM2-nonce-mn3-bs1024-dl0-ep1-sft-mix_qa_without_options-ep3
    # do
    #     echo "====== Evaluating $model_folder of $MODEL_NAME ======"
    #     model_path=$PROJECT_BASE_PATH/output/$MODEL_NAME/$model_folder
    #     /home/pj25000107/ku50001566/.local/bin/uv run python $PROJECT_BASE_PATH/src/evaluate.py \
    #         --model $model_path \
    #         --mode $MODE \
    #         --tokenizer $MODEL_NAME \
    #         --test-data $PROJECT_BASE_PATH/input/evaluate_data/qa/without_options/$eval_name/test.json \
    #         --score-on $SCORE_ON \
    #         --sample-num $SAMPLE_NUM \
    #         -o $model_path/evaluation$SUFFIX/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    # done
done