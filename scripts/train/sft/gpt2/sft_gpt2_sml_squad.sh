PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-$HOME/projects/knowledge_decoupling}"
SCRIPT_PATH=$PROJECT_BASE_PATH/scripts/train/sft

# CONFIG_NAME=$1
# INIT_MODEL=$2
# DATA_PATH=$3
# OUT_PATH=$4
# EPOCHS=$5

/bin/bash $SCRIPT_PATH/sft.sh \
   $PROJECT_BASE_PATH/output/gpt2/smolLM2/smolLM2_bs1024_dl0_ep1-ext_test_mix_ep3 \
   $PROJECT_BASE_PATH/output/gpt2/smolLM2/smolLM2_bs1024_dl0_ep1-ext_test_mix_ep3-sft_qa_test_wo_context \
   3
