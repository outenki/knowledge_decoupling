#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
# 0820
for model_name in gpt-mini gpt-medium gpt-large;do
    # echo "====== Evaluating untrained $model_name ======"
    # uv run python $BASE_PATH/src/agreement_evaluation.py \
    #     --model-path $BASE_PATH/output/$model_name/init_model \
    #     --val-data $BASE_PATH/data/evaluate_data/agreement_evaluate_data.json \
    #     -o $BASE_PATH/output/"$model_name"/init_model

    echo "====== Evaluating $model_name ======"
    for ep in 3 6; do
        uv run python $BASE_PATH/src/agreement_evaluation.py \
            --model-path $BASE_PATH/output/$model_name/wikitext-103-ep${ep} \
            --val-data $BASE_PATH/data/evaluate_data/agreement_evaluate_data.json \
            -o $BASE_PATH/output/"$model_name"/wikitext-103-ep${ep}
    done
done
