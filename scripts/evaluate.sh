#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling

# 0831
model_name=gpt-large
# eval_name=verb_agreement
# eval_name=qa_arc_challenge
# eval_name=qa_arc_easy
# eval_name=fce
# eval_name=qa_boolq
# eval_name=qa_qasc
for eval_name in verb_agreement fce_5gram qa_arc_easy qa_arc_challenge qa_boolq qa_boolq_psg qa_qasc; do
# for eval_name in qa_boolq; do
    echo
    echo "============ $eval_name ============"

    # echo "====== Evaluating hugging face gpt2 ======"
    # python $BASE_PATH/src/evaluate.py \
    #     --model-path gpt2 \
    #     --val-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
    #     -o $BASE_PATH/output/0830/gpt-large/gpt2/$eval_name

    for data_name in smolLM2-nonce-bs1024-dl1_020_000-ep3 smolLM2-nonce-bs1024-dl4_520_000-ep1 smolLM2-nonce-bs1024-dl0-ep1 smolLM2-bs1024-dl0-ep1;do
        echo
        echo "====== Evaluating $data_name ======"
        uv run python $BASE_PATH/src/evaluate.py \
            --speedup \
            --model-path $BASE_PATH/output/$model_name/smolLM2/$data_name \
            --val-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
            -o $BASE_PATH/output/$model_name/smolLM2/$data_name/$eval_name
    done
done