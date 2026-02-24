#! /bin/bash
# For extensive pretraining
INPUT_PATH=$HOME/projects/knowledge_decoupling/input/evaluate_data/unformated/
OUTPUT_PATH=$HOME/projects/knowledge_decoupling/input/tokenized/gpt2/sft

for dn in \
    mintaka_multihop \
    metaqa_1hop \
    metaqa_2hop \
    metaqa_3hop \
    qa_arc_challenge \
    qa_arc_easy \
    qa_qasc \
    squad_v2_ctxt_answerable;
do
    echo
    echo ">>>>>> $dn"
    uv run python ./tokenize_dataset_from_json.py \
        --tokenizer gpt2 \
        -mp \
        --input-path $INPUT_PATH/$dn/test.json \
        --output-path $OUTPUT_PATH/$dn/test
done
