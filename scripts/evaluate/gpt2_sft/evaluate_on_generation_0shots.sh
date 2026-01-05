#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling

MODEL_NAME=gpt2
SCORE_ON=generation
FEWSHOTS=0
SAMPLE_NUM=1000
MODE="simple"
SUFFIX="_20251222"

for eval_name in qa_arc_easy qa_arc_challenge qa_qasc squad_v2 squad_v2_ctxt; do
    echo
    echo "============ $eval_name ============"

    # hf model
    for model_folder in \
        hf-ext_train-ep1-sft-mix-ep3
    do
        echo "====== Evaluating $model_folder of $MODEL_NAME ======"
        model_path=$BASE_PATH/output/$MODEL_NAME/$model_folder
        /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
            --model $model_path \
            --mode $MODE \
            --tokenizer $MODEL_NAME \
            --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
            --score-on $SCORE_ON \
            --sample-num $SAMPLE_NUM \
            -o $model_path/evaluation$SUFFIX/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    done

    # trained by smolLM2
    for model_folder in \
        smolLM2/smolLM2-bs1024-dl0-ep1-ext_train-ep1-sft_mix_ep3
    do
        echo "====== Evaluating $model_folder of $MODEL_NAME ======"
        model_path=$BASE_PATH/output/$MODEL_NAME/smolLm2/$model_folder
        /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
            --model $model_path \
            --mode $MODE \
            --tokenizer $MODEL_NAME \
            --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
            --score-on $SCORE_ON \
            --sample-num $SAMPLE_NUM \
            -o $model_path/evaluation$SUFFIX/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    done
done