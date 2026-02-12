#! /bin/bash
# For extensive pretraining
INPUT_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/input/evaluate_data/unformated/
OUTPUT_PATH=/home/pj25000107/ku50001566/projects/knowledge_decoupling/input/tokenized/gpt2/ext/test_wo_answer

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
        --skip-answer \
        --input-path $INPUT_PATH/$dn/test.json \
        --output-path $OUTPUT_PATH/$dn
done
