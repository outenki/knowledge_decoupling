#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling

MODEL_NAME="Qwen/Qwen3-0.6B-Base"
SCORE_ON=generation
FEWSHOTS=0
MODE="simple"
SUFFIX="20251215"

# for eval_name in qa_boolq qa_boolq_ctxt squad_v2 squad_v2_ctxt; do
for eval_name in qa_arc_challenge; do
    echo
    echo "============ $eval_name ============"

    echo "====== Evaluating random $MODEL_NAME ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
        --model $BASE_PATH/output/$MODEL_NAME/random \
        --mode $MODE \
        --tokenizer $MODEL_NAME \
        --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
        --score-on $SCORE_ON \
        --sample-num 1000 \
        -o $BASE_PATH/output/$MODEL_NAME/random/evaluation_$SUFFIX/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name

    echo "====== Evaluating hugging face $MODEL_NAME ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
        --model $MODEL_NAME \
        --mode $MODE \
        --tokenizer $MODEL_NAME \
        --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
        --score-on $SCORE_ON \
        --sample-num 1000 \
        -o $BASE_PATH/output/$MODEL_NAME/hf/evaluation_$SUFFIX/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name

    echo "====== Evaluating $MODEL_NAME instruction ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
        --model "Qwen/Qwen3-0.6B" \
        --mode $MODE \
        --tokenizer $MODEL_NAME \
        --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
        --score-on $SCORE_ON \
        --sample-num 1000 \
        -o $BASE_PATH/output/$MODEL_NAME/hf-instruction/evaluation_$SUFFIX/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name

    echo "====== Evaluating hugging face $MODEL_NAME after SFT by mix ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
        --model $BASE_PATH/output/$MODEL_NAME/hf-sft/mix-e3 \
        --mode $MODE \
        --tokenizer $MODEL_NAME \
        --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
        --score-on $SCORE_ON \
        --sample-num 1000 \
        -o $BASE_PATH/output/$MODEL_NAME/hf-sft/mix-e3/evaluation_$SUFFIX/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name

    echo "====== Evaluating hugging face $MODEL_NAME after SFT by squad_v2_ctx ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
        --model $BASE_PATH/output/$MODEL_NAME/hf-sft/squad_v2_ctxt-e3 \
        --mode $MODE \
        --tokenizer $MODEL_NAME \
        --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
        --score-on $SCORE_ON \
        --sample-num 1000 \
        -o $BASE_PATH/output/$MODEL_NAME/hf-sft/squad_v2_ctxt-e3/evaluation_$SUFFIX/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name


    # for data_name in \
    #     smolLM2-bs1024-dl0-ep1 \
    #     smolLM2-nonce-mn3-bs1024-dl0-ep1
    # do
    #     echo
    #     echo "====== Evaluating $data_name ======"
    #     /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
    #         --model $BASE_PATH/output/$MODEL_NAME/$data_name \
    #         --mode $MODE \
    #         --tokenizer $MODEL_NAME \
    #         --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
    #         --score-on $SCORE_ON \
    #         --sample-num 1000 \
    #         -o $BASE_PATH/output/$MODEL_NAME/$data_name/evaluation_$SUFFIX/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    # done
done
