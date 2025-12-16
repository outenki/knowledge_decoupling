#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling

MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
SCORE_ON=generation
FEWSHOTS=0
SAMPLE_NUM=1000
MODE="simple"
SUFFIX="_20251127"

for eval_name in qa_arc_easy qa_arc_challenge qa_qasc squad_v2 squad_v2_ctxt; do
    echo
    echo "============ $eval_name ============"

    echo "====== Evaluating random $MODEL_NAME ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
        --model $BASE_PATH/output/$MODEL_NAME/random \
        --mode $MODE \
        --tokenizer $MODEL_NAME \
        --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
        --score-on $SCORE_ON \
        --sample-num $SAMPLE_NUM \
        -o $BASE_PATH/output/$MODEL_NAME/random/evaluation_$SUFFIX/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name

    echo "====== Evaluating hugging face $MODEL_NAME ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
        --model $MODEL_NAME \
        --mode $MODE \
        --tokenizer $MODEL_NAME \
        --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
        --score-on $SCORE_ON \
        --sample-num 1000 \
        -o $BASE_PATH/output/$MODEL_NAME/hf/evaluation/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name

    echo "====== Evaluating hugging face $MODEL_NAME after SFT by mix ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
        --model $BASE_PATH/output/$MODEL_NAME/hf-sft/mix-e3 \
        --mode $MODE \
        --tokenizer $MODEL_NAME \
        --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
        --score-on $SCORE_ON \
        --sample-num $SAMPLE_NUM \
        -o $BASE_PATH/output/$MODEL_NAME/hf-sft/mix-e3/evaluation_$SUFFIX/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name

    echo "====== Evaluating hugging face $MODEL_NAME after SFT by squad_v2_ctx ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
        --model $BASE_PATH/output/$MODEL_NAME/hf-sft/squad_v2_ctxt-e3 \
        --mode $MODE \
        --tokenizer $MODEL_NAME \
        --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
        --score-on $SCORE_ON \
        --sample-num $SAMPLE_NUM \
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
    #         --sample-num $SAMPLE_NUM \
    #         -o $BASE_PATH/output/$MODEL_NAME/$data_name/evaluation_$SUFFIX/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    # done
done