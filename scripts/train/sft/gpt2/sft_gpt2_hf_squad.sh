PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-/home/pj25000107/ku50001566/projects/knowledge_decoupling}"
SCRIPT_PATH=$PROJECT_BASE_PATH/scripts/train/sft

# CONFIG_NAME=$1
# INIT_MODEL=$2
# DATA_PATH=$3
# OUT_PATH=$4
# EPOCHS=$5
/bin/bash $SCRIPT_PATH/sft.sh \
    gpt2 \
    gpt2 \
    $PROJECT_BASE_PATH/input/tokenized/gpt2/sft/squad_v2_ctxt_ans \
    $PROJECT_BASE_PATH/output/gpt2/HuggingFace/hf-sft_squad_ans_ep3 \
    3

/bin/bash $SCRIPT_PATH/sft.sh \
    gpt2 \
    $PROJECT_BASE_PATH/output/gpt2/HuggingFace/hf-ext_test_ep3 \
    $PROJECT_BASE_PATH/input/tokenized/gpt2/sft/squad_v2_ctxt_ans \
    $PROJECT_BASE_PATH/output/gpt2/HuggingFace/hf-ext_test_ep3-sft_squad_ans_ep3 \
    3