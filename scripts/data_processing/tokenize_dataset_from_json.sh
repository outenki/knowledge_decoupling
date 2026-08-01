#! /bin/bash
# For extensive pretraining
# TOKENIZER=gpt2
# TOKENIZER_NAME="HuggingFaceTB/SmolLM2-135M"
# TOKENIZER_NAME="meta-llama/Llama-3.2-1B"
# TOKENIZER_NAME="allenai/OLMo-2-0425-1B"
# TOKENIZER_NAME="Qwen/Qwen2.5-0.5B"
# TOKENIZER_NAME="openai-community/gpt2"
# TOKENIZER=$PROJECT_BASE_PATH/output/$TOKENIZER_NAME/hf_full

INPUT_PATH=$PROJECT_BASE_PATH/input/evaluate_data/json
for TOKENIZER_NAME in meta-llama/Llama-3.2-1B Qwen/Qwen2.5-0.5B allenai/OLMo-2-0425-1B; do
    TOKENIZER=$TOKENIZER_NAME
    OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER_NAME/sft/concat
    for dn in \
        google_boolq_core_rnd \
        squadv2_core_rnd \
        triviaqa_rc_context_core_rnd
    do
        echo
        echo ">>>>>> $dn sft concat train"
        uv run python ./tokenize_dataset_from_json.py \
            -mp \
            --max-length 4096 \
            --tokenizer $TOKENIZER \
            --input-path $INPUT_PATH/$dn/train.json \
            --output-path $OUTPUT_PATH/$dn/train

        echo
        echo ">>>>>> $dn sft concat test"
        uv run python ./tokenize_dataset_from_json.py \
            -mp \
            --max-length 4096 \
            --tokenizer $TOKENIZER \
            --input-path $INPUT_PATH/$dn/test.json \
            --output-path $OUTPUT_PATH/$dn/test
    done
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


# INPUT_PATH=$PROJECT_BASE_PATH/data/ext
# OUTPUT_PATH=$PROJECT_BASE_PATH/input/tokenized/$TOKENIZER/ext/concat
# for dn in \
#     boolq
# do
#     echo
#     echo ">>>>>> $dn ext concat train"
#     uv run python ./tokenize_dataset_from_json.py \
#         --tokenizer $TOKENIZER \
#         --input-path $INPUT_PATH/$dn/train.json \
#         --output-path $OUTPUT_PATH/$dn/train
    
#     echo
#     echo ">>>>>> $dn ext concat test"
#     uv run python ./tokenize_dataset_from_json.py \
#         --tokenizer $TOKENIZER \
#         --input-path $INPUT_PATH/$dn/test.json \
#         --output-path $OUTPUT_PATH/$dn/test
# done

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
