#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling
for model_name in gpt-mini gpt-medium gpt-large;do
    for data_name in wikitext nonce;do
        #           10k   50k   100k   200k   300k   400k   500k
        for size in 10000 50000 100000 200000 300000 400000 500000;do
            uv run python agreement_evaluation.py \
                --model-path $BASE_PATH/output/$model_name/${data_name}_${size} \
                --model-type pt \
                --val-data $BASE_PATH/data/evaluate_data/agreement_evaluate_data.json \
                -o $BASE_PATH/output/"$model_name"/${data_name}_${size}
        done
    done
done
