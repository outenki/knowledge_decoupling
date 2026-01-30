#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-/home/pj25000107/ku50001566/projects/knowledge_decoupling}"

MODEL_NAME=gpt2
SCORE_ON=options
FEWSHOTS=0
SAMPLE_NUM=1000
MODE="simple"
SUFFIX="_20251215"

for eval_name in verb_agreement fce_5gram qa_arc_easy qa_arc_challenge qa_qasc qa_boolq qa_boolq_ctxt; do
    echo
    echo "============ $eval_name ============"

    # hf model
    for model_folder in \
        hf-ext_train-ep1-sft-mix-ep3
    do
        echo "====== Evaluating $model_folder of $MODEL_NAME ======"
        model_path=$PROJECT_BASE_PATH/output/$MODEL_NAME/$model_folder
        /home/pj25000107/ku50001566/.local/bin/uv run python $PROJECT_BASE_PATH/src/evaluate.py \
            --model $model_path \
            --mode $MODE \
            --tokenizer $MODEL_NAME \
            --test-data $PROJECT_BASE_PATH/input/evaluate_data/$eval_name/test.json \
            --score-on $SCORE_ON \
            --sample-num $SAMPLE_NUM \
            -o $model_path/evaluation$SUFFIX/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    done

    # trained by smolLM2
    for model_folder in \
        smolLM2/smolLM2-bs1024-dl0-ep1-ext_train-ep1-sft_mix_ep3
    do
        echo "====== Evaluating $model_folder of $MODEL_NAME ======"
        model_path=$PROJECT_BASE_PATH/output/$MODEL_NAME/smolLm2/$model_folder
        /home/pj25000107/ku50001566/.local/bin/uv run python $PROJECT_BASE_PATH/src/evaluate.py \
            --model $model_path \
            --mode $MODE \
            --tokenizer $MODEL_NAME \
            --test-data $PROJECT_BASE_PATH/input/evaluate_data/$eval_name/test.json \
            --score-on $SCORE_ON \
            --sample-num $SAMPLE_NUM \
            -o $model_path/evaluation$SUFFIX/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    done
done
