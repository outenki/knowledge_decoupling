#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-/home/pj25000107/ku50001566/projects/knowledge_decoupling}"
SCRIPT_PATH=$PROJECT_BASE_PATH/src
DATA_PATH=$PROJECT_BASE_PATH/input/tokenized/gpt2/ext

CONFIG_NAME="gpt2"
INIT_MODEL=$PROJECT_BASE_PATH/output/gpt2/smolLM2/smolLM2_bs1024_dl0_ep1
EPOCHS=3


echo "====== continue training ${INIT_MODEL} ======"
start_time=$(date +"%s")
echo "start time: $(date -d @"$start_time" +"%D %T")"

split="test"
uv run python "$SCRIPT_PATH"/train.py \
    --speedup \
    -cn $CONFIG_NAME \
    -im "$INIT_MODEL" \
    -dp "$DATA_PATH"/metaqa_1hop/$split \
    -dl 0 \
    -dp "$DATA_PATH"/metaqa_2hop/$split \
    -dl 0 \
    -dp "$DATA_PATH"/metaqa_3hop/$split \
    -dl 0 \
    -dp "$DATA_PATH"/mintaka_multihop/$split \
    -dl 0 \
    -dp "$DATA_PATH"/cwq/$split \
    -dl 0 \
    -dp "$DATA_PATH"/qa_arc_challenge/$split \
    -dl 0 \
    -dp "$DATA_PATH"/qa_arc_easy/$split \
    -dl 0 \
    -dp "$DATA_PATH"/qa_qasc/$split \
    -dl 0 \
    -dp "$DATA_PATH"/squad_v2_ctxt_answerable/$split \
    -dl 0 \
    -e $EPOCHS \
    -o "$INIT_MODEL"-ext_qa_${split}_ep${EPOCHS}

end_time=$(date +"%s")
echo "end time: $(date -d @"$end_time" +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
