#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling

# 0831
model_name=gpt2

score_on=options
# score_on=generation

FEWSHOTS=0
# eval_name=verb_agreement
# eval_name=qa_arc_challenge
# eval_name=qa_arc_easy
# eval_name=fce
# eval_name=qa_boolq
# eval_name=qa_qasc
# for eval_name in qa_qasc qa_boolq qa_boolq_psg; do
# for eval_name in qa_boolq_psg squad_v2; do
for eval_name in verb_agreement fce_5gram qa_arc_easy qa_arc_challenge qa_qasc qa_boolq qa_boolq_psg squad_v2 squad_v2_ctxt; do
    echo
    echo "============ $eval_name ============"

    # echo "====== Evaluating hugging face gpt2 ======"
    # /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
    #     --model-path gpt2 \
    #     --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
    #     --example-data $BASE_PATH/input/evaluate_data/$eval_name/examples.json \
    #     --score-on $score_on \
    #     --sample-num 1000 \
    #     -o $BASE_PATH/output/gpt2/gpt2_hf/evaluation_fewshots/score_on_${score_on}/$eval_name

        # smolLM2-nonce-bs1024-dl1_020_000-ep3 \
        # smolLM2-nonce-bs1024-dl4_520_000-ep1 \
        # smolLM2-nonce-bs1024-dl0-ep1 \
        # smolLM2-bs1024-dl1_020_000-ep3 \
        # smolLM2-bs1024-dl4_520_000-ep1 \
        # smolLM2-bs1024-dl0-ep1 \
        # smolLM2-nonce-mn3-bs1024-dl0-ep1 \
        # smolLM2-ox3000-bs1024-dl0-ep3
    for data_name in \
        squad_v2_ctxt-dl0-ep3-sft_gpt2
    do
        echo
        echo "====== Evaluating $data_name ======"
        # /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
        #     --model-path $BASE_PATH/output/$model_name/smolLM2/$data_name/init_model \
        #     --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
        #     --example-data $BASE_PATH/input/evaluate_data/$eval_name/examples.json \
        #     --score-on $score_on \
        #     --sample-num 1000 \
        #     -o $BASE_PATH/output/$model_name/smolLM2/$data_name/init_model/evaluation_fewshots/score_on_${score_on}/$eval_name
        /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
            --model-path $BASE_PATH/output/$model_name/smolLM2/$data_name \
            --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
            --score-on $score_on \
            --sample-num 1000 \
            -o $BASE_PATH/output/$model_name/smolLM2/$data_name/evaluation/fewshots_$FEWSHOTS/score_on_${score_on}/$eval_name
    done
done
