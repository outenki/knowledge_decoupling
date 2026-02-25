#!/bin/bash
PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH=$PROJECT_BASE_PATH/src

INIT_MODEL=$1
OUT_PATH=$2
EPOCHS=$3


echo "====== training on qa_w_context_test ======"
start_time=$(date +"%s")
echo "start time: $(date -d @"$start_time" +"%D %T")"


uv run python "$SCRIPT_PATH"/train.py \
    --speedup \
    -pad \
    -cn gpt2 \
    -im "$INIT_MODEL" \
    -dp "$PROJECT_BASE_PATH"/input/tokenized/gpt2/sft/squad_v2_ctxt_answerable/test \
    -dl 0 \
    -e "$EPOCHS" \
    -o "$OUT_PATH"

end_time=$(date +"%s")
echo "end time: $(date -d @"$end_time" +"%D %T")"
diff_sec=$(( end_time - start_time ))
hours=$(( diff_sec / 3600 ))
minutes=$(( (diff_sec % 3600) / 60 ))
seconds=$(( diff_sec % 60 ))
echo "Total time cost: ${hours}:${minutes}:${seconds}"
