#! /bin/bash
# For extensive pretraining
# TOKENIZER=gpt2
TOKENIZER=Qwen/Qwen3.5-0.8B-Base

    # commonsense_qa \
    # arc_challenge \
    # arc_easy \
    # google_re_long \
    # google_re_short \
    # squad_v2_answerable \
    # qasc
INPUT_PATH=$PROJECT_BASE_PATH/data/sft
OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/sft/concat
for dn in \
    squad_v2_answerable
do
    echo
    echo ">>>>>> $dn sft concat train"
    uv run python ./tokenize_dataset_from_json.py \
        -mp \
        --tokenizer $TOKENIZER \
        --input-path $INPUT_PATH/$dn/train.json \
        --output-path $OUTPUT_PATH/$dn/train

    echo
    echo ">>>>>> $dn ext concat test"
    uv run python ./tokenize_dataset_from_json.py \
        -mp \
        --tokenizer $TOKENIZER \
        --input-path $INPUT_PATH/$dn/test.json \
        --output-path $OUTPUT_PATH/$dn/test
done

INPUT_PATH=$PROJECT_BASE_PATH/data/sft
OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/sft/chat_template
for dn in \
    squad_v2_answerable
do
    echo
    echo ">>>>>> $dn sft chat_template train"
    uv run python ./tokenize_dataset_from_json.py \
        -ct \
        -mp \
        --tokenizer $TOKENIZER \
        --input-path $INPUT_PATH/$dn/train.json \
        --output-path $OUTPUT_PATH/$dn/train

    echo
    echo ">>>>>> $dn sft chat_template test"
    uv run python ./tokenize_dataset_from_json.py \
        -ct \
        -mp \
        --tokenizer $TOKENIZER \
        --input-path $INPUT_PATH/$dn/test.json \
        --output-path $OUTPUT_PATH/$dn/test
done


INPUT_PATH=$PROJECT_BASE_PATH/data/ext
OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/ext/concat
for dn in \
    squad_v2_answerable
do
    echo
    echo ">>>>>> $dn ext concat train"
    uv run python ./tokenize_dataset_from_json.py \
        --tokenizer $TOKENIZER \
        --input-path $INPUT_PATH/$dn/train.json \
        --output-path $OUTPUT_PATH/$dn/train
    
    echo
    echo ">>>>>> $dn ext concat test"
    uv run python ./tokenize_dataset_from_json.py \
        --tokenizer $TOKENIZER \
        --input-path $INPUT_PATH/$dn/test.json \
        --output-path $OUTPUT_PATH/$dn/test
done

INPUT_PATH=$PROJECT_BASE_PATH/data/ext
OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/ext/chat_template
for dn in \
    squad_v2_answerable
do
    echo
    echo ">>>>>> $dn ext chat_template train"
    uv run python ./tokenize_dataset_from_json.py \
        -ct \
        --tokenizer $TOKENIZER \
        --input-path $INPUT_PATH/$dn/train.json \
        --output-path $OUTPUT_PATH/$dn/train
    echo
    echo ">>>>>> $dn ext chat_template test"
    uv run python ./tokenize_dataset_from_json.py \
        -ct \
        --tokenizer $TOKENIZER \
        --input-path $INPUT_PATH/$dn/test.json \
        --output-path $OUTPUT_PATH/$dn/test
done


    # verb_agreement \
    # fce \
    # fce_3gram \
    # fce_5gram \
    # boolq_ctxt \
    # cflx_clasheval \
    # cflx_nq_swap \
    # cwq \
    # google_re_long \
    # google_re_short \
    # google_re_no_context \
    # metaqa_1hop \
    # metaqa_2hop \
    # metaqa_3hop \
    # mintaka \
    # mintaka_multihop \
    # squad_v2 \
    # squad_v2_answerable \
    # squad_v2_wo_ctxt
# for dn in \
#     commonsense_qa \
#     arc_challenge \
#     arc_easy \
#     qasc
# do
#     echo
#     echo ">>>>>> $dn"
#     uv run python ./tokenize_dataset_from_json.py \
#         -mp \
#         --tokenizer $TOKENIZER \
#         --input-path $INPUT_PATH/$dn/train.json \
#         --output-path $OUTPUT_PATH/$dn/train
#     uv run python ./tokenize_dataset_from_json.py \
#         -mp \
#         --tokenizer $TOKENIZER \
#         --input-path $INPUT_PATH/$dn/test.json \
#         --output-path $OUTPUT_PATH/$dn/test
#     uv run python ./tokenize_dataset_from_json.py \
#         -sa \
#         -mp \
#         --tokenizer $TOKENIZER \
#         --input-path $INPUT_PATH/$dn/train.json \
#         --output-path $OUTPUT_PATH/"$dn"_que/train
#     uv run python ./tokenize_dataset_from_json.py \
#         -sa \
#         -mp \
#         --tokenizer $TOKENIZER \
#         --input-path $INPUT_PATH/$dn/test.json \
#         --output-path $OUTPUT_PATH/"$dn"_que/test
# done
