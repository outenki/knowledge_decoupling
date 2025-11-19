#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling

FEWSHOTS=0
# MODEL_PATH=gpt2
# MODEL_PATH="meta-llama/Llama-3.2-1B"
MODEL_PATH="Qwen/Qwen3-0.6B-Base"

EVAL_NAME=verb_agreement 
echo
echo "============ $EVAL_NAME ============"
/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate_test.py \
    --model-path $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --test-data $BASE_PATH/input/evaluate_data/$EVAL_NAME/test.json \
    --score-on options \
    -o $BASE_PATH/output/test_model/evaluation/score_on_options/0_shot/$EVAL_NAME

EVAL_NAME=qa_arc_easy
echo
echo "============ $EVAL_NAME ============"
/home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate_test.py \
    --model-path $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --test-data $BASE_PATH/input/evaluate_data/$EVAL_NAME/test.json \
    --example-data $BASE_PATH/input/evaluate_data/$EVAL_NAME/examples.json \
    --score-on options \
    --sample-num 100 \
    -o $BASE_PATH/output/test_model/evaluation/score_on_generation/few_shots/$EVAL_NAME
