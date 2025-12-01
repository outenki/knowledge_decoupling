#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling

MODEL_NAME=gpt2
SCORE_ON=options
FEWSHOTS=0
SAMPLE_NUM=1000
MODE="simple"
SUFFIX="_20251127_${MODE}"

for eval_name in verb_agreement fce_5gram qa_arc_easy qa_arc_challenge qa_qasc qa_boolq qa_boolq_ctxt; do
    echo
    echo "============ $eval_name ============"
    # echo "====== Evaluating hugging face $MODEL_NAME ======"
    # /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
    #     --model $MODEL_NAME \
    #     --mode $MODE \
    #     --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
    #     --score-on $SCORE_ON \
    #     --sample-num $SAMPLE_NUM \
    #     -o $BASE_PATH/output/$MODEL_NAME/${MODEL_NAME}-hf/evaluation${SUFFIX}/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name

    # for model in \
    #     gpt2-random \
    #     gpt2-sft_squad_v2_ctxt-ep3 \
    #     gpt2-sft_mix-ep3
    # do
    #     echo "====== Evaluating hugging face $model ======"
    #     /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
    #         --model $BASE_PATH/output/$MODEL_NAME/$model \
    #         --mode $MODE \
    #         --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
    #         --score-on $SCORE_ON \
    #         --sample-num $SAMPLE_NUM \
    #         -o $BASE_PATH/output/$MODEL_NAME/$model/evaluation${SUFFIX}/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    # done
    #     smolLM2-nonce-bs1024-dl0-ep1 \
    #     smolLM2-nonce-mn3-bs1024-dl0-ep1 \
    #     smolLM2-ox3000-bs1024-dl0-ep3 \
    #     smolLM2-ox3000-bs1024-dl0-ep3-sft_squad_v2_ctxt-ep3 \
    #     smolLM2-ox3000-bs1024-dl0-ep3-sft_mix-ep3 \
    #     smolLM2-bs1024-dl0-ep1 \
    #     smolLM2-bs1024-dl0-ep1-sft_squad_v2_ctxt-ep3 \
    #     smolLM2-bs1024-dl0-ep1-sft_mix-ep3
        # smolLM2-nonce-mn3-bs1024-dl0-ep1-sft_squad_v2_ctxt-e3
    for model in \
        smolLM2-nonce-bs1024-dl0-ep1-sft_squad_v2_ctxt-e3
    do
        echo
        echo "====== Evaluating $model ======"
        /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
            --model $BASE_PATH/output/$MODEL_NAME/smolLM2/$model \
            --mode $MODE \
            --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
            --score-on $SCORE_ON \
            --sample-num $SAMPLE_NUM \
            -o $BASE_PATH/output/$MODEL_NAME/smolLM2/$model/evaluation${SUFFIX}/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    done
done
