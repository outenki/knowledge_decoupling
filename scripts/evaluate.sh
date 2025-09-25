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
# for eval_name in verb_agreement qa_arc_challenge qa_arc_easy fce qa_boolq qa_qasc; do
for eval_name in qa_boolq_psg; do
    echo
    echo "============ $eval_name ============"
    echo "====== Evaluating hugging face gpt2 ======"
    python $BASE_PATH/src/evaluate.py \
        --model-path gpt2 \
        --val-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
        -o $BASE_PATH/output/0830/gpt-large/gpt2/$eval_name
    for data_name in init_model wikimedia-bs1024-ep1 wikimedia-nonce-bs1024-ep1 wikimedia-nonce-bs1024-ep3;do
        echo
        echo "====== Evaluating $data_name ======"
        python $BASE_PATH/src/evaluate.py \
            --model-path $BASE_PATH/output/0830/$model_name/$data_name \
            --val-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
            -o $BASE_PATH/output/0830/"$model_name"/$data_name/$eval_name
    done
done
# 0820
# for model_name in gpt-mini gpt-medium gpt-large;do
#     echo "====== Evaluating untrained $model_name ======"
#     uv run python $BASE_PATH/src/evaluate.py \
#         --model-path $BASE_PATH/output/$model_name/init_model \
#         --val-data $BASE_PATH/data/evaluate_data/agreement_evaluate_data.json \
#         -o $BASE_PATH/output/"$model_name"/init_model

#     echo "====== Evaluating $model_name ======"
#     uv run python $BASE_PATH/src/evaluate.py \
#         --model-path $BASE_PATH/output/$model_name/wikitext-103-ep3 \
#         --val-data $BASE_PATH/data/evaluate_data/agreement_evaluate_data.json \
#         -o $BASE_PATH/output/"$model_name"/wikitext-103-ep3
# done

# for model_name in gpt-mini gpt-medium gpt-large;do
#     echo "====== Evaluating untrained $model_name ======"
#     uv run python $BASE_PATH/src/evaluate.py \
#         --model-path $BASE_PATH/output/$model_name/init_model \
#         --val-data $BASE_PATH/data/evaluate_data/agreement_evaluate_data.json \
#         -o $BASE_PATH/output/"$model_name"/init_model

#     for epoch in 3 6; do
#         for data_name in wikitext nonce;do
#             #           10k   50k   100k   200k   300k   400k   500k
#             for size in 10000 50000 100000 200000 300000 400000 500000; do
#                 echo "====== Evaluating $model_name trained with $data_name $size epoch ${epoch}======"
#                 uv run python $BASE_PATH/src/evaluate.py \
#                     --model-path $BASE_PATH/output/$model_name/${data_name}_${size}_${epoch} \
#                     --val-data $BASE_PATH/data/evaluate_data/agreement_evaluate_data.json \
#                     -o $BASE_PATH/output/"$model_name"/${data_name}_${size}_${epoch}
#             done
#         done
#     done
# done
