#! /bin/bash
# For extensive pretraining
# TOKENIZER=gpt2
# TOKENIZER_NAME="meta-llama/Llama-3.2-1B"
# TOKENIZER_NAME="HuggingFaceTB/SmolLM2-135M"
TOKENIZER_NAME="openai-community/gpt2"
TOKENIZER=$PROJECT_BASE_PATH/output/$TOKENIZER_NAME/hf_full

    # commonsense_qa \
    # arc_challenge \
    # arc_easy \
    # google_re_long \
    # google_re_short \
    # squad_v2_answerable \
    # qasc
INPUT_PATH=$PROJECT_BASE_PATH/data/sft
OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER_NAME/sft/concat
for dn in \
    boolq
do
    echo
    echo ">>>>>> $dn sft concat train"
    uv run python ./tokenize_dataset_from_json.py \
        -mp \
        --tokenizer $TOKENIZER \
        --input-path $INPUT_PATH/$dn/train.json \
        --output-path $OUTPUT_PATH/$dn/train

    echo
    echo ">>>>>> $dn sft concat test"
    uv run python ./tokenize_dataset_from_json.py \
        -mp \
        --tokenizer $TOKENIZER \
        --input-path $INPUT_PATH/$dn/test.json \
        --output-path $OUTPUT_PATH/$dn/test
done

# INPUT_PATH=$PROJECT_BASE_PATH/data/sft
# OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/sft/chat_template
# for dn in \
#     squad_v2_answerable
# do
#     echo
#     echo ">>>>>> $dn sft chat_template train"
#     uv run python ./tokenize_dataset_from_json.py \
#         -ct \
#         -mp \
#         --tokenizer $TOKENIZER \
#         --input-path $INPUT_PATH/$dn/train.json \
#         --output-path $OUTPUT_PATH/$dn/train

#     echo
#     echo ">>>>>> $dn sft chat_template test"
#     uv run python ./tokenize_dataset_from_json.py \
#         -ct \
#         -mp \
#         --tokenizer $TOKENIZER \
#         --input-path $INPUT_PATH/$dn/test.json \
#         --output-path $OUTPUT_PATH/$dn/test
# done


INPUT_PATH=$PROJECT_BASE_PATH/data/ext
OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/ext/concat
for dn in \
    boolq
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

# INPUT_PATH=$PROJECT_BASE_PATH/data/ext
# OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/ext/chat_template
# for dn in \
#     squad_v2_answerable
# do
#     echo
#     echo ">>>>>> $dn ext chat_template train"
#     uv run python ./tokenize_dataset_from_json.py \
#         -ct \
#         --tokenizer $TOKENIZER \
#         --input-path $INPUT_PATH/$dn/train.json \
#         --output-path $OUTPUT_PATH/$dn/train
#     echo
#     echo ">>>>>> $dn ext chat_template test"
#     uv run python ./tokenize_dataset_from_json.py \
#         -ct \
#         --tokenizer $TOKENIZER \
#         --input-path $INPUT_PATH/$dn/test.json \
#         --output-path $OUTPUT_PATH/$dn/test
# done
