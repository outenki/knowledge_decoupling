#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"

MODEL_NAME=gpt2
SCORE_ON=generation
FEWSHOTS=0
SAMPLE_NUM=1000
MODE="full"
SUFFIX="_$MODE"


    # squad_v2_ctxt_answerable \
    # cwq \
for eval_name in \
    qa_arc_easy \
    qa_arc_challenge \
    metaqa_1hop \
    metaqa_2hop \
    metaqa_3hop \
    mintaka_multihop \
    qa_qasc
do
    echo
    echo "============ $eval_name ============"
    echo "====== Evaluating Pretrained $MODEL_NAME ======"
    # model_path=gpt2
    # python $PROJECT_BASE_PATH/src/evaluate.py \
    #     --model $model_path \
    #     --mode $MODE \
    #     --tokenizer $MODEL_NAME \
    #     --test-data $PROJECT_BASE_PATH/input/evaluate_data/unformated/$eval_name/test.json \
    #     --score-on $SCORE_ON \
    #     --sample-num $SAMPLE_NUM \
    #     -o $PROJECT_BASE_PATH/output/$MODEL_NAME/HuggingFace/hf/evaluation$SUFFIX/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name

    for model_folder in \
        smolLM2/smolLM2_bs1024_dl0_ep1-ext_test_mix_ep3-sft_qa_test_wo_context \
        random/rnd-sft_qa_test_wo_context
    do
        echo "====== Evaluating $model_folder of $MODEL_NAME ======"
        model_path=$PROJECT_BASE_PATH/output/$MODEL_NAME/$model_folder
        uv run python $PROJECT_BASE_PATH/src/evaluate.py \
            --model $model_path \
            --mode $MODE \
            --tokenizer $MODEL_NAME \
            --test-data $PROJECT_BASE_PATH/input/evaluate_data/unformated/$eval_name/test.json \
            --score-on $SCORE_ON \
            --sample-num $SAMPLE_NUM \
            -o $model_path/evaluation$SUFFIX/${SCORE_ON}/${FEWSHOTS}_shots/$eval_name
    done
done
