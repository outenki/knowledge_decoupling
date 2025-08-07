#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
for model_name in gpt-mini gpt-medium gpt-large;do
    echo "====== Evaluating untrained $model_name ======"
    uv run python $BASE_PATH/src/agreement_evaluation.py \
        --model-path $BASE_PATH/output/$model_name/init_model \
        --val-data $BASE_PATH/data/evaluate_data/agreement_evaluate_data.json \
        -o $BASE_PATH/output/"$model_name"/init_model

    for data_name in wikitext nonce;do
        #           10k   50k   100k   200k   300k   400k   500k
        for size in 100000 200000 300000 400000 500000;do
            echo "====== Evaluating $model_name trained with $data_name $size ======"
            uv run python $BASE_PATH/src/agreement_evaluation.py \
                --model-path $BASE_PATH/output/$model_name/${data_name}_${size} \
                --val-data $BASE_PATH/data/evaluate_data/agreement_evaluate_data.json \
                -o $BASE_PATH/output/"$model_name"/${data_name}_${size}
        done
    done
done
