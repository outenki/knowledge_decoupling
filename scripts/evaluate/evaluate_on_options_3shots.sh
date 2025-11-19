#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling

MODEL_NAME=gpt2
SCORE_ON=options
MODEL_HF=gpt2
FEWSHOTS=3

for eval_name in verb_agreement fce_5gram qa_arc_easy qa_arc_challenge qa_qasc qa_boolq qa_boolq_psg; do
    echo
    echo "============ $eval_name ============"

    echo "====== Evaluating random $MODEL_HF ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
        --model-path $BASE_PATH/output/$MODEL_NAME/${MODEL_NAME}_random \
        --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
        --score-on $SCORE_ON \
        --sample-num 1000 \
        -o $BASE_PATH/output/$MODEL_NAME/${MODEL_NAME}_random/evaluation/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name

    echo "====== Evaluating hugging face $MODEL_HF ======"
    /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
        --model-path $MODEL_HF \
        --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
        --score-on $SCORE_ON \
        --sample-num 1000 \
        -o $BASE_PATH/output/$MODEL_NAME/gpt2_hf/evaluation/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name


    for data_name in \
        smolLM2-nonce-bs1024-dl0-ep1 \
        smolLM2-ox3000-bs1024-dl0-ep3 \
        squad_v2_ctxt-dl0-ep3-sft_smolLM2-ox3000-bs1024-dl0-ep3 \
        smolLM2-bs1024-dl0-ep1 \
        squad_v2_ctxt-dl0-ep3-sft_smolLM2-bs1024-dl0-ep1 \
        squad_v2_ctxt-dl0-ep3-sft_gpt2
    do
        echo
        echo "====== Evaluating $data_name ======"
        /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
            --model-path $BASE_PATH/output/$MODEL_NAME/smolLM2/$data_name \
            --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
            --score-on $SCORE_ON \
            --sample-num 1000 \
            -o $BASE_PATH/output/$MODEL_NAME/smolLM2/$data_name/evaluation/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    done
done
