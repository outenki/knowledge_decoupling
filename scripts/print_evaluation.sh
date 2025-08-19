#!/bin/bash
BASE_PATH="/home/pj25000107/ku50001566/projects/knowledge_decoupling/output"

for model in gpt-mini gpt-medium gpt-large;do
    for data in wikitext nonce;do
        for epoch in 3 6;do
            echo "==========${model}-${data}-EPOCH_${epoch}========="
            for size in 50000 100000 200000 300000 400000 500000;do
                cat $BASE_PATH/$model/${data}_${size}_${epoch}/evaluation_summary.json|grep accu|cut -d":" -f2|cut -d"," -f1
            done
        done
    done
done