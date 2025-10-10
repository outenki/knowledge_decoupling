#!/bin/bash
# BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
BASE_PATH=/Users/ou/Developer/projects/knowledge_decoupling

# 0831
model_name=gpt-large
# eval_name=verb_agreement
# eval_name=qa_arc_challenge
# eval_name=qa_arc_easy
# eval_name=fce
# eval_name=qa_boolq
# eval_name=qa_qasc
for eval_name in verb_agreement fce fce_3gram qa_arc_easy qa_arc_challenge qa_boolq qa_boolq_psg qa_qasc; do
# for eval_name in fce_3gram; do
    echo
    echo "============ $eval_name ============"

    # echo "====== Evaluating hugging face gpt2 ======"
    # python $BASE_PATH/src/evaluate.py \
    #     --model-path gpt2 \
    #     --val-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
    #     -o $BASE_PATH/output/0830/gpt-large/gpt2/$eval_name
    for data_name in wikimedia-nonce-bs1024-ep4;do
    # for data_name in init_model_0 init_model_1 init_model_2 init_model_3 init_model_4;do
        echo
        echo "====== Evaluating $data_name ======"
        python $BASE_PATH/src/evaluate.py \
            --model-path $BASE_PATH/output/1001/$model_name/$data_name \
            --val-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
            -o $BASE_PATH/output/1001/$model_name/$data_name/$eval_name
    done
done