#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling

FEWSHOTS=0
MODEL_PATH=gpt2
for eval_name in verb_agreement qa_arc_easy; do
    echo
    echo "============ $eval_name ============"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate_batch.py \
        --model-path $MODEL_PATH \
        --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
        --example-data $BASE_PATH/input/evaluate_data/$eval_name/examples.json \
        --score-on $score_on \
        --sample-num 1000 \
        -o $BASE_PATH/output/test_model/evaluation/score_on_options/$eval_name
done

eval_name=squad_v2_ctxt
echo
echo "============ $eval_name ============"
/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate_batch.py \
    --model-path $MODEL_PATH \
    --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
    --example-data $BASE_PATH/input/evaluate_data/$eval_name/examples.json \
    --score-on $score_on \
    --sample-num 1000 \
    -o $BASE_PATH/output/test_model/evaluation/score_on_generation/$eval_name
