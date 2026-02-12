#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-/home/pj25000107/ku50001566/projects/knowledge_decoupling}"
SCRIPT_PATH=$PROJECT_BASE_PATH/src
OUT_PATH=$PROJECT_BASE_PATH/output
DATA_PATH=$PROJECT_BASE_PATH/input/tokenized/gpt2/ext/test_wo_answer

CONFIG_NAME="gpt2"
INIT_MODEL=$PROJECT_BASE_PATH/output/gpt2/nonce/smolLM2_135M_sents_shuffled_bs1024_ep3
EPOCHS=3


echo "====== continue training ${INIT_MODEL} ======"
start_time=$(date +"%s")
echo "start time: $(date -d @$start_time +"%D %T")"


uv run python $SCRIPT_PATH/train.py \
    --speedup \
    -cn $CONFIG_NAME \
    -im $INIT_MODEL \
    -dp "$DATA_PATH"/metaqa_1hop \
    -dp "$DATA_PATH"/metaqa_2hop \
    -dp "$DATA_PATH"/metaqa_3hop \
    -dp "$DATA_PATH"/mintaka_multihop \
    -dp "$DATA_PATH"/qa_arc_challenge \
    -dp "$DATA_PATH"/qa_arc_easy \
    -dp "$DATA_PATH"/qa_qasc \
    -dp "$DATA_PATH"/squad_v2_ctxt_answerable \
    -e $EPOCHS \
    -o $OUT_PATH/$CONFIG_NAME/nonce/smolLM2_135M_sents_shuffled_bs1024_ep3-ext_test_mix_ep${EPOCHS}

end_time=$(date +"%s")
echo "end time: $(date -d @$end_time +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
