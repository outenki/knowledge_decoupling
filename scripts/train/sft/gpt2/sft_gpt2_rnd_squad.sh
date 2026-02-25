PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH=$PROJECT_BASE_PATH/scripts/train/sft/gpt2

# CONFIG_NAME=$1
# INIT_MODEL=$2
# DATA_PATH=$3
# OUT_PATH=$4
# EPOCHS=$5

/bin/bash $SCRIPT_PATH/sft.sh \
    $PROJECT_BASE_PATH/output/gpt2/random/rnd \
    $PROJECT_BASE_PATH/output/gpt2/random/rnd-sft_qa_test_wo_context \
    3
