#!/bin/bash
BASE_PATH=/Users/ou/projects/knowledge_decoupling

MODEL_NAME=gpt2
SCORE_ON=generation
FEWSHOTS=0
SAMPLE_NUM=1000
MODE="simple"
SUFFIX=""


for eval_name in unformated/squad_v2_ctxt; do
    echo
    echo "============ $eval_name ============"
    echo "====== Evaluating Pretrained $MODEL_NAME ======"
    model_path=gpt2
    uv run python $BASE_PATH/src/evaluate.py \
        --model $model_path \
        --mode $MODE \
        --tokenizer $MODEL_NAME \
        --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
        --score-on $SCORE_ON \
        --sample-num $SAMPLE_NUM \
        -o $BASE_PATH/output/$MODEL_NAME/HuggingFace/hf/evaluation$SUFFIX/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name

    for model_folder in \
        HuggingFace/hf-ext_test_ep3 \
        HuggingFace/hf-sft_squad_ans_ep3 \
        HuggingFace/hf-ext_test_ep3-sft_squad_ans_ep3 \
        smolLM2/smolLM2_bs1024_dl0_ep1 \
        smolLM2/smolLM2_bs1024_dl0_ep1-ext_test_squad_answerable_ep3 \
        smolLM2/smolLM2_bs1024_dl0_ep1-sft_squad_ans_ep3 \
        smolLM2/smolLM2_bs1024_dl0_ep1-ext_test_squad_answerable_ep3-sft_squad_ans_ep3 \
        nonce/smolLM2_nonce_mn3_bs1024_dl0_ep1 \
        nonce/smolLM2_nonce_mn3_bs1024_dl0_ep1-ext_test_nonce_ep3 \
        nonce/smolLM2_nonce_mn3_bs1024_dl0_ep1-sft_squad_ans_ep3 \
        nonce/smolLM2_nonce_mn3_bs1024_dl0_ep1-ext_test_nonce_ep3-sft_squad_ans_ep3
    do
        echo "====== Evaluating $model_folder of $MODEL_NAME ======"
        model_path=$BASE_PATH/output/$MODEL_NAME/$model_folder
        uv run python $BASE_PATH/src/evaluate.py \
            --model $model_path \
            --mode $MODE \
            --tokenizer $MODEL_NAME \
            --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
            --score-on $SCORE_ON \
            --sample-num $SAMPLE_NUM \
            -o $model_path/evaluation$SUFFIX/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    done
done