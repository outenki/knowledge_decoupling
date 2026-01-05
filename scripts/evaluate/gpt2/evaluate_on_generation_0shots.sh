#!/bin/bash
BASE_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling

MODEL_NAME=gpt2
SCORE_ON=generation
FEWSHOTS=0
SAMPLE_NUM=1000
MODE="simple"
SUFFIX="_20251222"

for eval_name in qa_arc_easy qa_arc_challenge qa_qasc qa_boolq_ctxt squad_v2_ctxt; do
    echo
    echo "============ $eval_name ============"

    # hf model
    # for model_folder in \
    #     hf-sft-qa_boolq_ctxt-ep3
    # do
    #     echo "====== Evaluating $model_folder of $MODEL_NAME ======"
    #     model_path=$BASE_PATH/output/$MODEL_NAME/$model_folder
    #     /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
    #         --model $model_path \
    #         --mode $MODE \
    #         --tokenizer $MODEL_NAME \
    #         --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
    #         --score-on $SCORE_ON \
    #         --sample-num $SAMPLE_NUM \
    #         -o $model_path/evaluation$SUFFIX/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    # done

    # trained by smolLM2
    for model_folder in \
        smolLM2-bs1024-dl0-ep1-ext_train-ep1-sft_mix_ep3
    do
        echo "====== Evaluating $model_folder of $MODEL_NAME ======"
        model_path=$BASE_PATH/output/$MODEL_NAME/smolLM2/$model_folder
        /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
            --model $model_path \
            --mode $MODE \
            --tokenizer $MODEL_NAME \
            --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
            --score-on $SCORE_ON \
            --sample-num $SAMPLE_NUM \
            -o $model_path/evaluation$SUFFIX/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    done

    # trained by nonce
    # for model_folder in \
    #     smolLM2-nonce-mn3-bs1024-dl0-ep1-sft-qa_boolq_ctxt-ep3
    # do
    #     echo "====== Evaluating $model_folder of $MODEL_NAME ======"
    #     model_path=$BASE_PATH/output/$MODEL_NAME/nonce/$model_folder
    #     /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
    #         --model $model_path \
    #         --mode $MODE \
    #         --tokenizer $MODEL_NAME \
    #         --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
    #         --score-on $SCORE_ON \
    #         --sample-num $SAMPLE_NUM \
    #         -o $model_path/evaluation$SUFFIX/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    # done
done

    # echo "====== Evaluating random $MODEL_NAME ======"
    # /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
    #     --model $BASE_PATH/output/$MODEL_NAME/random \
    #     --mode $MODE \
    #     --tokenizer $MODEL_NAME \
    #     --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
    #     --score-on $SCORE_ON \
    #     --sample-num $SAMPLE_NUM \
    #     -o $BASE_PATH/output/$MODEL_NAME/random/evaluation$SUFFIX/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name

    # echo "====== Evaluating hugging face $MODEL_NAME ======"
    # /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
    #     --model $MODEL_NAME \
    #     --mode $MODE \
    #     --tokenizer $MODEL_NAME \
    #     --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
    #     --score-on $SCORE_ON \
    #     --sample-num 1000 \
    #     -o $BASE_PATH/output/$MODEL_NAME/hf/evaluation$SUFFIX/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name

    # echo "====== Evaluating hugging face $MODEL_NAME after SFT by mix ======"
    # /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
    #     --model $BASE_PATH/output/$MODEL_NAME/hf-sft/mix-e3 \
    #     --mode $MODE \
    #     --tokenizer $MODEL_NAME \
    #     --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
    #     --score-on $SCORE_ON \
    #     --sample-num $SAMPLE_NUM \
    #     -o $BASE_PATH/output/$MODEL_NAME/hf-sft/mix-e3/evaluation$SUFFIX/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name

    # echo "====== Evaluating hugging face $MODEL_NAME after SFT by squad_v2_ctx ======"
    # /home/pj25000107/ku50001566/.local/bin/uv run python $BASE_PATH/src/evaluate.py \
    #     --model $BASE_PATH/output/$MODEL_NAME/hf-sft/squad_v2_ctxt-e3 \
    #     --mode $MODE \
    #     --tokenizer $MODEL_NAME \
    #     --test-data $BASE_PATH/input/evaluate_data/$eval_name/test.json \
    #     --score-on $SCORE_ON \
    #     --sample-num $SAMPLE_NUM \
    #     -o $BASE_PATH/output/$MODEL_NAME/hf-sft/squad_v2_ctxt-e3/evaluation$SUFFIX/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name



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
    #         -o $BASE_PATH/output/$MODEL_NAME/$data_name/evaluation$SUFFIX/$score_on_${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    # done