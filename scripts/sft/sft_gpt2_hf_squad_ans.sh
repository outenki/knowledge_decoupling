# CONFIG_NAME=$1
# INIT_MODEL=$2
# DATA=$3
# OUTPUT_NAME=$4 -> CONFIG_NAME/OUTPUT_NAME
# EPOCHS=$5
sh sft.sh \
    gpt2 \
    /Users/ou/projects/knowledge_decoupling/output/gpt2/HuggingFace/hf-ext_test_squad_answerable_ep3 \
    /Users/ou/projects/knowledge_decoupling/input/tokenized/gpt2/sft/squad_v2_ctxt_1000_unans \
    /Users/ou/projects/knowledge_decoupling/output/gpt2/HuggingFace/hf-ext_test_squad_answerable_ep3-sft_squad_v2_ctxt_1000_unans \
    3